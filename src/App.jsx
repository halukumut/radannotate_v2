import { useState, useRef, useEffect, useCallback } from "react";

// ── Renk paleti ───────────────────────────────────────────────────────────
const C = {
  bg:"#0a0c10",surface:"#111318",surfaceAlt:"#161922",border:"#1e2230",
  accent:"#3b7ff5",accentSoft:"#1e3a6e",
  success:"#22c55e",successSoft:"#0f2e1c",
  warning:"#f59e0b",warningSoft:"#2d1f06",
  danger:"#ef4444",dangerSoft:"#2d0f0f",
  text:"#e2e8f0",textMuted:"#64748b",textDim:"#374151",
  ai:"#a78bfa",aiSoft:"#1e1435",
  gt:"#34d399",gtSoft:"#0a2419",
  physician:"#fb923c",physicianSoft:"#2d1506",
};

// ── YOLO (cx cy w h) → canvas (x y w h sol-üst köşe) ─────────────────────
function yoloToRect(b) {
  return { x: b.cx - b.w / 2, y: b.cy - b.h / 2, w: b.w, h: b.h };
}

// ── IoU — iki rect arasında ───────────────────────────────────────────────
function iouRect(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  if (x2 <= x1 || y2 <= y1) return 0;
  const inter = (x2 - x1) * (y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter);
}

// Hekim bbox listesi vs GT bbox listesi → max IoU
function bestIou(physicianBoxes, gtYoloBoxes) {
  if (!physicianBoxes.length || !gtYoloBoxes.length) return null;
  let best = 0;
  for (const pb of physicianBoxes)
    for (const gb of gtYoloBoxes)
      best = Math.max(best, iouRect(pb, yoloToRect(gb)));
  return +best.toFixed(4);
}

function formatTime(ms) {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function confColor(conf) {
  if (conf >= 0.80) return C.success;
  if (conf >= 0.55) return C.warning;
  return C.danger;
}

// ── Canvas bbox handle sabitleri ──────────────────────────────────────────
const HR = 6;

function getHandles(b, W, H) {
  const bx = b.x * W, by = b.y * H, bw = b.w * W, bh = b.h * H;
  return [
    { id:"tl", cx:bx,          cy:by,          cursor:"nwse-resize" },
    { id:"tm", cx:bx+bw/2,     cy:by,          cursor:"ns-resize"   },
    { id:"tr", cx:bx+bw,       cy:by,          cursor:"nesw-resize" },
    { id:"ml", cx:bx,          cy:by+bh/2,     cursor:"ew-resize"   },
    { id:"mr", cx:bx+bw,       cy:by+bh/2,     cursor:"ew-resize"   },
    { id:"bl", cx:bx,          cy:by+bh,       cursor:"nesw-resize" },
    { id:"bm", cx:bx+bw/2,     cy:by+bh,       cursor:"ns-resize"   },
    { id:"br", cx:bx+bw,       cy:by+bh,       cursor:"nwse-resize" },
  ];
}

function applyResize(b, hid, dx, dy) {
  let {x,y,w,h} = b;
  if(hid.includes("l")){x+=dx;w-=dx;} if(hid.includes("r")){w+=dx;}
  if(hid.includes("t")){y+=dy;h-=dy;} if(hid.includes("b")){h+=dy;}
  if(w<0.01){if(hid.includes("l"))x=b.x+b.w-0.01;w=0.01;}
  if(h<0.01){if(hid.includes("t"))y=b.y+b.h-0.01;h=0.01;}
  x=Math.max(0,x); y=Math.max(0,y);
  w=Math.min(1-x,w); h=Math.min(1-y,h);
  return {...b,x,y,w,h};
}

function hitHandle(b,px,py,W,H){
  for(const h of getHandles(b,W,H))
    if(Math.abs(px-h.cx)<=HR+2 && Math.abs(py-h.cy)<=HR+2) return h;
  return null;
}

function hitBox(b,px,py,W,H){
  return px>=b.x*W && px<=(b.x+b.w)*W && py>=b.y*H && py<=(b.y+b.h)*H;
}

// ── Küçük UI ─────────────────────────────────────────────────────────────
function Badge({color,children}){
  return <span style={{background:color+"22",border:`1px solid ${color}44`,color,borderRadius:4,padding:"2px 8px",fontSize:11,fontWeight:600,letterSpacing:"0.04em",textTransform:"uppercase"}}>{children}</span>;
}

function ProgressBar({value,color=C.accent}){
  return <div style={{background:C.border,borderRadius:99,height:6,overflow:"hidden"}}><div style={{width:`${Math.min(100,value*100)}%`,height:"100%",background:color,borderRadius:99,transition:"width 0.4s ease"}}/></div>;
}

function Stat({label,value,sub,color=C.text}){
  return <div style={{textAlign:"center"}}>
    <div style={{fontSize:26,fontWeight:700,color,fontFamily:"monospace",letterSpacing:"-0.02em"}}>{value}</div>
    {sub&&<div style={{fontSize:11,color:C.textMuted,marginTop:2}}>{sub}</div>}
    <div style={{fontSize:12,color:C.textMuted,marginTop:4}}>{label}</div>
  </div>;
}

// ── AnnotationCanvas ──────────────────────────────────────────────────────
function AnnotationCanvas({imageData,boxes,selectedIdx,onBoxesChange,onSelectIdx,showGt,showPred}){
  const ASPECT_RATIO = 700 / 900; // width/height ratio (wider box for portrait images)
  const [canvasSize, setCanvasSize] = useState({w:700,h:900});
  const canvasRef = useRef(null);
  const imgRef    = useRef(null);
  const interRef  = useRef({mode:"idle"});
  const imgDimsRef = useRef({w:1,h:1}); // original image dimensions
  const displayDimsRef = useRef({ox:0, oy:0, w:700, h:467}); // scaled display dims + offset
  const boxesRef  = useRef(boxes);
  const selRef    = useRef(selectedIdx);
  const canvasSizeRef = useRef(canvasSize);
  useEffect(()=>{canvasSizeRef.current=canvasSize;},[canvasSize]);
  useEffect(()=>{boxesRef.current=boxes;},[boxes]);
  useEffect(()=>{selRef.current=selectedIdx;},[selectedIdx]);

  // ResizeObserver - responsive to container width
  useEffect(()=>{
    const ro = new ResizeObserver(([e])=>{
      const w=e.contentRect.width;
      const maxW=Math.min(w,700); // cap at 700px for desktop
      setCanvasSize({w:maxW, h:Math.round(maxW/ASPECT_RATIO)});
    });
    if(canvasRef.current?.parentElement) ro.observe(canvasRef.current.parentElement);
    return ()=>ro.disconnect();
  },[]);

  // Store original image dimensions and compute letterboxed display size
  useEffect(()=>{
    if(!imgRef.current) return;
    const iw = imgRef.current.naturalWidth || imgRef.current.width;
    const ih = imgRef.current.naturalHeight || imgRef.current.height;
    if(iw>0 && ih>0) {
      imgDimsRef.current = {w:iw, h:ih};
      // Compute letterboxed dimensions to fit current canvas size
      const {w: CANVAS_W, h: CANVAS_H} = canvasSizeRef.current;
      const scale = Math.min(CANVAS_W / iw, CANVAS_H / ih);
      const displayW = Math.round(iw * scale);
      const displayH = Math.round(ih * scale);
      const ox = Math.round((CANVAS_W - displayW) / 2);
      const oy = Math.round((CANVAS_H - displayH) / 2);
      displayDimsRef.current = {ox, oy, w: displayW, h: displayH};
    }
  },[imageData?.url,canvasSize]);

  // Görsel yükle
  useEffect(()=>{
    imgRef.current=null;
    if(!imageData?.url) return;
    const img=new Image();
    img.onload=()=>{imgRef.current=img; draw();};
    img.onerror=()=>{imgRef.current=null; draw();};
    img.src=imageData.url;
  },[imageData?.url]); // eslint-disable-line

  // Klavye sil
  useEffect(()=>{
    const onKey=(e)=>{
      if((e.key==="Delete"||e.key==="Backspace")&&selRef.current!==null){
        onBoxesChange(boxesRef.current.filter((_,i)=>i!==selRef.current));
        onSelectIdx(null);
      }
    };
    window.addEventListener("keydown",onKey);
    return ()=>window.removeEventListener("keydown",onKey);
  },[onBoxesChange,onSelectIdx]);

  const draw = useCallback(()=>{
    const canvas=canvasRef.current; if(!canvas) return;
    const ctx=canvas.getContext("2d");
    const {w: CANVAS_W, h: CANVAS_H} = canvasSizeRef.current;
    canvas.width=CANVAS_W; canvas.height=CANVAS_H;

    // Arka plan
    ctx.fillStyle="#060810"; ctx.fillRect(0,0,CANVAS_W,CANVAS_H);

    // Görsel - draw letterboxed in responsive box
    const {ox, oy, w: displayW, h: displayH} = displayDimsRef.current;
    if(imgRef.current){
      ctx.drawImage(imgRef.current, ox, oy, displayW, displayH);
    } else {
      // Grid placeholder
      ctx.strokeStyle="#1a2030"; ctx.lineWidth=1;
      for(let x=0;x<CANVAS_W;x+=32){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,CANVAS_H);ctx.stroke();}
      for(let y=0;y<CANVAS_H;y+=32){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(CANVAS_W,y);ctx.stroke();}
      ctx.fillStyle=C.textDim; ctx.font="12px monospace"; ctx.textAlign="center";
      ctx.fillText(imageData?.file??"Görsel bekleniyor…",CANVAS_W/2,CANVAS_H/2);
      ctx.textAlign="left";
    }

    // Bbox çizim yardımcısı
    // YOLO coords are ratios of original image, convert to display pixels
    const {w: origW, h: origH} = imgDimsRef.current;
    const scale = displayW / origW; // scale factor from original to display
    const drawBox=(rect,color,dash,label,alpha=1)=>{
      const px = ox + rect.x * displayW, py = oy + rect.y * displayH, pw = rect.w * displayW, ph = rect.h * displayH;
      ctx.globalAlpha=alpha;
      ctx.strokeStyle=color; ctx.lineWidth=2;
      if(dash) ctx.setLineDash(dash); else ctx.setLineDash([]);
      ctx.strokeRect(px,py,pw,ph);
      ctx.fillStyle=color+"20"; ctx.fillRect(px,py,pw,ph);
      ctx.setLineDash([]);
      if(label){
        const tw=ctx.measureText(label).width+8;
        ctx.fillStyle=color+"ee";
        ctx.fillRect(px,py-16,tw,16);
        ctx.fillStyle="#fff"; ctx.font="bold 10px monospace";
        ctx.fillText(label,px+4,py-4);
      }
      ctx.globalAlpha=1;
    };

    // GT bbox'ları
    if(showGt && imageData?.gt){
      imageData.gt.forEach((gb,i)=>{
        drawBox(yoloToRect(gb), C.gt, [5,4],
          imageData.gt.length>1?`GT #${i+1}`:"GT");
      });
    }

    // Pred bbox'ları (sadece AI oturumda, showPred açıksa) — metin etiketi olmadan
    if(showPred && imageData?.pred){
      imageData.pred.forEach((pb,i)=>{
        const col=confColor(pb.conf??1);
        drawBox(yoloToRect(pb), col, [8,4],
          `AI ${Math.round((pb.conf??1)*100)}%${imageData.pred.length>1?` #${i+1}`:""}`,
          0.9);
      });
    }

    // Hekim bbox'ları
    boxesRef.current.forEach((b,i)=>{
      const sel=i===selRef.current;
      drawBox(b, C.physician, null,
        boxesRef.current.length>1?`#${i+1}`:"Hekim",
        sel?1:0.8);
      if(sel){
        getHandles(b,w,h).forEach(hd=>{
          ctx.fillStyle=C.physician; ctx.strokeStyle="#060810"; ctx.lineWidth=1.5;
          ctx.beginPath(); ctx.rect(hd.cx-HR,hd.cy-HR,HR*2,HR*2); ctx.fill(); ctx.stroke();
        });
      }
    });

    // Çizim önizlemesi
    const inter=interRef.current;
    if(inter.mode==="drawing"&&inter.drawStart&&inter.drawCurrent){
      const ds=inter.drawStart, dc=inter.drawCurrent;
      const x=Math.min(ds.x,dc.x), y=Math.min(ds.y,dc.y);
      const bw=Math.abs(dc.x-ds.x), bh=Math.abs(dc.y-ds.y);
      if(bw>0.005&&bh>0.005) drawBox({x,y,w:bw,h:bh},C.physician,[5,3],null);
    }
  },[imageData,showGt,showPred,canvasSize]);

  useEffect(()=>{draw();},[draw,boxes,selectedIdx,canvasSize]);

  const getPos=(e)=>{
    const rect=canvasRef.current.getBoundingClientRect();
    const cx=e.touches?e.touches[0].clientX:e.clientX;
    const cy=e.touches?e.touches[0].clientY:e.clientY;
    const px=cx-rect.left, py=cy-rect.top;
    const {ox, oy, w: displayW, h: displayH} = displayDimsRef.current;
    const {w: origW, h: origH} = imgDimsRef.current;
    // Convert screen coords to original image ratio space (0-1)
    const nx = (px - ox) / displayW;
    const ny = (py - oy) / displayH;
    return {px, py, nx, ny};
  };

  const getCursor=(px,py)=>{
    const {ox, oy, w: displayW, h: displayH} = displayDimsRef.current;
    const bxs=boxesRef.current; const sel=selRef.current;
    if(sel!==null){
      const hd=hitHandle(bxs[sel],px-ox,py-oy,displayW,displayH); if(hd) return hd.cursor;
      if(hitBox(bxs[sel],px-ox,py-oy,displayW,displayH)) return "move";
    }
    for(let i=bxs.length-1;i>=0;i--)
      if(hitBox(bxs[i],px-ox,py-oy,displayW,displayH)) return "move";
    return "crosshair";
  };

  const onDown=(e)=>{
    if(!imageData) return;
    e.preventDefault(); canvasRef.current?.focus();
    const {px,py,nx,ny}=getPos(e);
    const {ox, oy, w: displayW, h: displayH} = displayDimsRef.current;
    const bxs=boxesRef.current; const sel=selRef.current;
    const inter=interRef.current;

    if(sel!==null){
      const hd=hitHandle(bxs[sel],px-ox,py-oy,displayW,displayH);
      if(hd){inter.mode="resize";inter.startPx=px;inter.startPy=py;inter.origBox={...bxs[sel]};inter.handleId=hd.id;return;}
      if(hitBox(bxs[sel],px-ox,py-oy,displayW,displayH)){inter.mode="move";inter.startPx=px;inter.startPy=py;inter.origBox={...bxs[sel]};return;}
    }
    for(let i=bxs.length-1;i>=0;i--){
      if(hitBox(bxs[i],px-ox,py-oy,displayW,displayH)){
        onSelectIdx(i); selRef.current=i;
        inter.mode="move";inter.startPx=px;inter.startPy=py;inter.origBox={...bxs[i]};return;
      }
    }
    onSelectIdx(null); selRef.current=null;
    inter.mode="drawing"; inter.drawStart={x:nx,y:ny}; inter.drawCurrent={x:nx,y:ny};
  };

  const onMove=(e)=>{
    const {px,py,nx,ny}=getPos(e);
    const {ox, oy, w: displayW, h: displayH} = displayDimsRef.current;
    const inter=interRef.current;
    canvasRef.current.style.cursor=getCursor(px,py);

    if(inter.mode==="drawing"){inter.drawCurrent={x:nx,y:ny};draw();}
    else if(inter.mode==="move"&&inter.origBox){
      const dx=(px-inter.startPx)/displayW, dy=(py-inter.startPy)/displayH;
      const sel=selRef.current; if(sel===null) return;
      const ob=inter.origBox;
      onBoxesChange(boxesRef.current.map((b,i)=>i===sel?{...b,x:Math.max(0,Math.min(1-ob.w,ob.x+dx)),y:Math.max(0,Math.min(1-ob.h,ob.y+dy))}:b));
    }
    else if(inter.mode==="resize"&&inter.origBox){
      const dx=(px-inter.startPx)/displayW, dy=(py-inter.startPy)/displayH;
      const sel=selRef.current; if(sel===null) return;
      onBoxesChange(boxesRef.current.map((b,i)=>i===sel?applyResize(inter.origBox,inter.handleId,dx,dy):b));
    }
  };

  const onUp=(e)=>{
    const {nx,ny}=getPos(e);
    const inter=interRef.current;
    if(inter.mode==="drawing"&&inter.drawStart){
      const ds=inter.drawStart;
      const x=Math.min(ds.x,nx), y=Math.min(ds.y,ny);
      const bw=Math.abs(nx-ds.x), bh=Math.abs(ny-ds.y);
      if(bw>0.02&&bh>0.02){
        const nb=[...boxesRef.current,{id:Date.now(),x,y,w:bw,h:bh}];
        onBoxesChange(nb); onSelectIdx(nb.length-1);
      }
    }
    inter.mode="idle"; inter.drawStart=null; inter.drawCurrent=null; inter.origBox=null;
    draw();
  };

  const {w,h}=canvasSize;
  return <canvas ref={canvasRef} tabIndex={0} width={w} height={h}
    style={{width:w,height:h,cursor:"crosshair",borderRadius:8,border:`1px solid ${C.border}`,display:"block",userSelect:"none",outline:"none"}}
    onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
    onTouchStart={onDown} onTouchMove={onMove} onTouchEnd={onUp}
  />;
}

// ── Pred bbox paneli ──────────────────────────────────────────────────────
function PredPanel({pred}){
  if(!pred?.length) return (
    <div style={{background:C.aiSoft,border:`1px solid ${C.ai}22`,borderRadius:8,padding:"10px 14px",fontSize:12,color:C.textMuted}}>
      Bu görsel için model tahmini yok.
    </div>
  );
  return (
    <div style={{background:C.aiSoft,border:`1px solid ${C.ai}33`,borderRadius:8,padding:"12px 14px"}}>
      <div style={{fontSize:11,color:C.ai,textTransform:"uppercase",letterSpacing:"0.06em",marginBottom:10}}>AI Tahminleri</div>
      <div style={{display:"flex",flexDirection:"column",gap:7}}>
        {pred.map((pb,i)=>{
          const col=confColor(pb.conf??1);
          const pct=Math.round((pb.conf??1)*100);
          return (
            <div key={i} style={{display:"flex",alignItems:"center",gap:10}}>
              <span style={{fontSize:11,color:C.textDim,fontFamily:"monospace",width:22}}>#{i+1}</span>
              <div style={{flex:1,background:C.border,borderRadius:99,height:6,overflow:"hidden"}}>
                <div style={{width:`${pct}%`,height:"100%",background:col,borderRadius:99,transition:"width 0.3s"}}/>
              </div>
              <span style={{fontSize:12,fontWeight:700,color:col,fontFamily:"monospace"}}>%{pct} eminlik skoru</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Rapor ─────────────────────────────────────────────────────────────────
function ReportView({results,mode,onReset}){
  const avg=(arr)=>arr.length?arr.reduce((a,b)=>a+b,0)/arr.length:0;
  const times=results.map(r=>r.time);
  const ious=results.map(r=>r.iou).filter(v=>v!==null);
  const correct=results.filter(r=>r.labelCorrect).length;
  const avgTime=avg(times);
  const totalTime=times.reduce((a,b)=>a+b,0);
  const avgIou=avg(ious);
  const accuracy=correct/results.length;

  const [saveStatus,setSaveStatus]=useState("pending");
  const [savedUrl,setSavedUrl]=useState(null);

  useEffect(()=>{
    setSaveStatus("saving");
    const payload={sessionId:`session_${Date.now()}`,mode,physicianId:"demo-user",completedAt:new Date().toISOString(),results};
    fetch("/api/stats",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)})
      .then(r=>r.json())
      .then(d=>setSaveStatus(d.ok?"saved":"error"))
      .catch(()=>{
        const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});
        setSavedUrl(URL.createObjectURL(blob));
        setSaveStatus("download");
      });
  },[]); // eslint-disable-line

  const iouColor=avgIou>0.7?C.success:avgIou>0.5?C.warning:C.danger;
  const accColor=accuracy>0.8?C.success:accuracy>0.6?C.warning:C.danger;

  return (
    <div style={{padding:"24px 0"}}>
      <div style={{marginBottom:20,display:"flex",alignItems:"center",justifyContent:"space-between"}}>
        <div>
          <div style={{fontSize:20,fontWeight:700,color:C.text,marginBottom:4}}>Oturum Tamamlandı</div>
          <div style={{fontSize:13,color:C.textMuted}}>{mode==="solo"?"Solo Hekim":"Hekim + AI"} · {results.length} görsel</div>
        </div>
        <Badge color={mode==="solo"?C.physician:C.ai}>{mode==="solo"?"Solo":"AI Destekli"}</Badge>
      </div>

      {saveStatus==="saving"&&<div style={{background:C.accentSoft,border:`1px solid ${C.accent}44`,borderRadius:8,padding:"10px 14px",marginBottom:14,fontSize:12,color:C.accent}}>Kaydediliyor...</div>}
      {saveStatus==="saved"&&<div style={{background:C.successSoft,border:`1px solid ${C.success}44`,borderRadius:8,padding:"10px 14px",marginBottom:14,fontSize:12,color:C.success}}>✓ Kaydedildi — <code>/api/stats</code></div>}
      {saveStatus==="error"&&<div style={{background:C.dangerSoft,border:`1px solid ${C.danger}44`,borderRadius:8,padding:"10px 14px",marginBottom:14,fontSize:12,color:C.danger}}>Kayıt başarısız</div>}
      {saveStatus==="download"&&savedUrl&&(
        <div style={{background:C.warningSoft,border:`1px solid ${C.warning}44`,borderRadius:8,padding:"10px 14px",marginBottom:14,fontSize:12,color:C.warning,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <span>Sunucuya ulaşılamadı — JSON olarak indirin</span>
          <a href={savedUrl} download={`radannotate_${mode}_${Date.now()}.json`} style={{color:C.warning,fontWeight:600,textDecoration:"none",border:`1px solid ${C.warning}55`,borderRadius:5,padding:"3px 10px"}}>İndir</a>
        </div>
      )}

      {/* 4 metrik kart */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10,marginBottom:20}}>
        {[
          {label:"Ort. Süre / görsel",value:formatTime(Math.round(avgTime)),sub:`Min ${formatTime(Math.min(...times))} · Max ${formatTime(Math.max(...times))}`,color:C.accent},
          {label:"Toplam Süre",value:formatTime(totalTime),sub:`${results.length} görsel`,color:C.accent},
          {label:"Doğruluk",value:`${Math.round(accuracy*100)}%`,sub:`${correct}/${results.length}`,color:accColor},
          {label:"Ort. IoU",value:ious.length?avgIou.toFixed(2):"—",sub:"Hekim bbox vs GT",color:iouColor},
        ].map(s=>(
          <div key={s.label} style={{background:C.surfaceAlt,border:`1px solid ${C.border}`,borderRadius:12,padding:"14px 10px"}}>
            <Stat {...s}/>
          </div>
        ))}
      </div>

      {/* Görsel tablo */}
      <div style={{marginBottom:20}}>
        <div style={{fontSize:12,fontWeight:600,color:C.textMuted,marginBottom:10,textTransform:"uppercase",letterSpacing:"0.06em"}}>Görsel Bazlı Sonuçlar</div>
        <div style={{display:"flex",flexDirection:"column",gap:5}}>
          {results.map((r,i)=>(
            <div key={i} style={{background:C.surfaceAlt,border:`1px solid ${C.border}`,borderRadius:7,padding:"8px 12px",display:"grid",gridTemplateColumns:"26px 1fr 68px 80px 88px",alignItems:"center",gap:8}}>
              <span style={{fontSize:11,color:C.textDim,fontFamily:"monospace"}}>#{String(i+1).padStart(2,"0")}</span>
              <div style={{fontSize:11,color:C.textMuted,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{r.stem}</div>
              <div style={{textAlign:"right",fontSize:12,color:C.accent,fontFamily:"monospace"}}>{formatTime(r.time)}</div>
              <div style={{textAlign:"right",fontSize:12,fontFamily:"monospace",color:r.iou!==null?(r.iou>0.5?C.success:C.danger):C.textDim}}>
                {r.iou!==null?`${r.iou.toFixed(2)} IoU`:"— boş"}
              </div>
              <div style={{textAlign:"right"}}><Badge color={r.labelCorrect?C.success:C.danger}>{r.labelCorrect?"DOĞRU":"YANLIŞ"}</Badge></div>
            </div>
          ))}
        </div>
      </div>

      {/* Süre bar */}
      <div style={{background:C.surfaceAlt,border:`1px solid ${C.border}`,borderRadius:12,padding:14,marginBottom:18}}>
        <div style={{fontSize:11,color:C.textMuted,marginBottom:10,textTransform:"uppercase",letterSpacing:"0.06em"}}>Görsel başına süre</div>
        <div style={{display:"flex",alignItems:"flex-end",gap:2,height:72}}>
          {times.map((t,i)=>(
            <div key={i} style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",gap:2}}>
              <div style={{width:"100%",height:(t/Math.max(...times))*60,background:C.accent+"99",borderRadius:"2px 2px 0 0"}}/>
              <div style={{fontSize:8,color:C.textDim,fontFamily:"monospace"}}>{i+1}</div>
            </div>
          ))}
        </div>
      </div>

      <button onClick={onReset} style={{width:"100%",padding:12,background:C.accentSoft,border:`1px solid ${C.accent}44`,borderRadius:8,color:C.accent,fontSize:14,fontWeight:600,cursor:"pointer"}}>
        Yeni Oturum Başlat
      </button>
    </div>
  );
}

// ── Ana uygulama ──────────────────────────────────────────────────────────
export default function App(){
  const [phase,    setPhase]    = useState("home");
  const [mode,     setMode]     = useState(null);
  const [manifest, setManifest] = useState(null);
  const [loadErr,  setLoadErr]  = useState(null);
  const [sessionImgs, setSessionImgs] = useState([]); // aktif set (A veya B)
  const [imgIdx,   setImgIdx]   = useState(0);
  const [results,  setResults]  = useState([]);
  const [boxes,    setBoxes]    = useState([]);
  const [selIdx,   setSelIdx]   = useState(null);
  const [startTime,setStartTime]= useState(null);
  const [showGt,   setShowGt]   = useState(false);
  const [showPred, setShowPred] = useState(true);

  const imageData = sessionImgs[imgIdx] ?? null;

  // manifest.json yükle
  const loadManifest=useCallback(async()=>{
    setPhase("loading"); setLoadErr(null);
    try{
      const res=await fetch("/data/manifest.json");
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      const data=await res.json();
      if(!data.images?.length) throw new Error("manifest boş");
      setManifest(data);
      setPhase("home");
    }catch(err){
      setLoadErr(err.message);
      setPhase("home");
    }
  },[]);

  useEffect(()=>{loadManifest();},[loadManifest]);

  const startSession=(m)=>{
    if (!manifest) return;
    // Solo → Set A, AI destekli → Set B
    const targetSet = m === "solo" ? "A" : "B";
    const sessionImages = manifest.images.filter(img => img.setId === targetSet);
    setSessionImgs(sessionImages);
    setMode(m); setImgIdx(0); setResults([]); setBoxes([]); setSelIdx(null);
    setStartTime(Date.now()); setShowGt(false); setShowPred(m==="ai");
    setPhase("session");
  };

  const submit=()=>{
    const elapsed=Date.now()-startTime;
    const gtBoxes=imageData?.gt??[];
    const iouScore=bestIou(boxes,gtBoxes);
    const hasGt=gtBoxes.length>0;

    // 4 durum doğruluk mantığı:
    // GT var  + hekim işaretledi → IoU > 0.3 ise doğru
    // GT var  + hekim boş bıraktı → yanlış (kaçırdı)
    // GT yok  + hekim boş bıraktı → doğru (true negative)
    // GT yok  + hekim işaretledi → yanlış (false positive)
    let labelCorrect;
    if (hasGt) {
      labelCorrect = boxes.length > 0 && iouScore !== null && iouScore > 0.3;
    } else {
      labelCorrect = boxes.length === 0;
    }

    const rec={
      stem:imageData?.stem,
      imageId:imageData?.id,
      setId:imageData?.setId,
      status:imageData?.status,
      gtBoxCount:gtBoxes.length,
      physicianBoxCount:boxes.length,
      labelCorrect,
      iou:iouScore,
      time:elapsed,
      boxes:boxes.map(b=>({x:+b.x.toFixed(4),y:+b.y.toFixed(4),w:+b.w.toFixed(4),h:+b.h.toFixed(4)})),
    };
    const newResults=[...results,rec];
    setResults(newResults);
    if(imgIdx+1>=sessionImgs.length){setPhase("report");}
    else{
      setImgIdx(i=>i+1); setBoxes([]); setSelIdx(null);
      setStartTime(Date.now()); setShowGt(false); setShowPred(mode==="ai");
    }
  };

  const reset=()=>{setPhase("home");setMode(null);setImgIdx(0);setResults([]);setBoxes([]);setSelIdx(null);setSessionImgs([]);};

  return (
    <div style={{minHeight:"100vh",background:C.bg,color:C.text,fontFamily:"'DM Sans','Segoe UI',sans-serif",paddingBottom:48}}>

      {/* Header */}
      <div style={{borderBottom:`1px solid ${C.border}`,padding:"14px 24px",display:"flex",alignItems:"center",justifyContent:"space-between",background:C.surface,position:"sticky",top:0,zIndex:10}}>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <div style={{width:30,height:30,borderRadius:8,background:C.accentSoft,border:`1px solid ${C.accent}33`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:14}}>⊕</div>
          <div>
            <div style={{fontSize:14,fontWeight:700,letterSpacing:"-0.01em"}}>RadAnnotate</div>
            <div style={{fontSize:10,color:C.textMuted}}>Hekim Performans Platformu</div>
          </div>
        </div>
        {phase==="session"&&manifest&&(
          <div style={{display:"flex",alignItems:"center",gap:14}}>
            <Badge color={mode==="solo"?C.physician:C.ai}>{mode==="solo"?"Solo · Set A":"AI Destekli · Set B"}</Badge>
            <span style={{fontSize:12,color:C.textMuted}}>{imgIdx+1} / {sessionImgs.length}</span>
          </div>
        )}
      </div>

      <div style={{maxWidth:840,margin:"0 auto",padding:"0 8px"}}>

        {/* HOME / LOADING */}
        {(phase==="home"||phase==="loading")&&(
          <div style={{paddingTop:48}}>
            <div style={{textAlign:"center",marginBottom:40}}>
              <div style={{fontSize:28,fontWeight:800,letterSpacing:"-0.03em",marginBottom:8}}>Radyoloji Anotasyon Benchmarkı</div>
              <div style={{fontSize:14,color:C.textMuted,maxWidth:460,margin:"0 auto",lineHeight:1.7}}>
                Hekim solo vs. AI destekli performansını ölçer.
              </div>
              {manifest&&(
                <div style={{marginTop:14,display:"flex",gap:10,justifyContent:"center",flexWrap:"wrap"}}>
                  <Badge color={C.accent}>{manifest.totalImages} görsel</Badge>
                  <Badge color={C.physician}>Set A: {manifest.setA?.count} solo</Badge>
                  <Badge color={C.ai}>Set B: {manifest.setB?.count} AI destekli</Badge>
                  <Badge color={C.gt}>GT: {manifest.withGt}</Badge>
                  <Badge color={C.textMuted}>Negatif: {manifest.negatives}</Badge>
                </div>
              )}
            </div>

            {phase==="loading"&&<div style={{textAlign:"center",color:C.textMuted,padding:24}}>manifest.json yükleniyor...</div>}

            {loadErr&&(
              <div style={{background:C.dangerSoft,border:`1px solid ${C.danger}44`,borderRadius:10,padding:"14px 18px",marginBottom:24,fontSize:13,color:C.danger}}>
                <strong>Manifest yüklenemedi:</strong> {loadErr}
                <br/><span style={{fontSize:11,color:C.textMuted}}>
                  <code>node scripts/build-manifest.js</code> çalıştırıp <code>public/data/manifest.json</code>'un varlığını doğrulayın.
                </span>
              </div>
            )}

            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14,marginBottom:28,opacity:manifest?1:0.4,pointerEvents:manifest?"auto":"none"}}>
              {[
                {m:"solo",icon:"👁",title:"Oturum A — Solo",desc:"AI önerisi olmadan bağımsız etiketleme. Pred ve GT gizli.",features:["Görseller yüklemeden gösterilir","Bbox + süre kaydedilir","AI önerisi görünmez"],col:C.physician},
                {m:"ai",icon:"🤖",title:"Oturum B — AI Destekli",desc:"Model pred bbox'ları ve güven skoru görünür. Hekimin kararı esastır.",features:["Pred bbox + conf score gösterilir","Çoklu tahmin desteği","Model doğruluk: %94"],col:C.ai},
              ].map(({m,icon,title,desc,features,col})=>(
                <div key={m} onClick={()=>startSession(m)}
                  style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:14,padding:24,cursor:"pointer",transition:"border-color 0.2s",position:"relative",overflow:"hidden"}}
                  onMouseEnter={e=>e.currentTarget.style.borderColor=col+"88"}
                  onMouseLeave={e=>e.currentTarget.style.borderColor=C.border}
                >
                  <div style={{position:"absolute",top:-16,right:-16,width:100,height:100,borderRadius:"50%",background:col+"08",border:`1px solid ${col}22`}}/>
                  <div style={{fontSize:28,marginBottom:10}}>{icon}</div>
                  <div style={{fontSize:15,fontWeight:700,marginBottom:6}}>{title}</div>
                  <div style={{fontSize:12,color:C.textMuted,lineHeight:1.6,marginBottom:14}}>{desc}</div>
                  {features.map(f=>(
                    <div key={f} style={{fontSize:12,color:C.textMuted,display:"flex",alignItems:"center",gap:6,marginBottom:4}}>
                      <span style={{color:col,fontSize:9}}>●</span>{f}
                    </div>
                  ))}
                </div>
              ))}
            </div>

            <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:18}}>
              <div style={{fontSize:11,fontWeight:600,color:C.textMuted,marginBottom:12,textTransform:"uppercase",letterSpacing:"0.06em"}}>Ölçülen Metrikler</div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:14}}>
                {[{l:"Süre",d:"Görsel başına ms"},{l:"Toplam Süre",d:"Oturum toplam"},{l:"Doğruluk",d:"Karar vs GT"},{l:"IoU",d:"Bbox örtüşme 0–1"}].map(({l,d})=>(
                  <div key={l}><div style={{fontSize:13,fontWeight:600,color:C.text,marginBottom:3}}>{l}</div><div style={{fontSize:11,color:C.textMuted}}>{d}</div></div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* SESSION */}
        {phase==="session"&&imageData&&(
          <div style={{paddingTop:20}}>
            <div style={{marginBottom:14}}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
                <span style={{fontSize:12,color:C.textMuted}}>İlerleme</span>
                <span style={{fontSize:12,color:C.textMuted}}>{imgIdx}/{sessionImgs.length}</span>
              </div>
              <ProgressBar value={imgIdx/sessionImgs.length}/>
            </div>

            {mode==="ai"&&<div style={{marginBottom:12}}><PredPanel pred={imageData.pred}/></div>}

            {/* Top button bar */}
            <div style={{display:"flex",alignItems:"center",justifyContent:"flex-end",gap:10,marginBottom:14}}>
              {mode==="ai"&&(
                <button onClick={()=>setShowPred(v=>!v)}
                  style={{padding:"6px 14px",borderRadius:6,border:`1px solid ${C.ai}44`,background:showPred?C.aiSoft:"transparent",color:C.ai,fontSize:12,cursor:"pointer",fontWeight:500}}>
                  {showPred?"Pred Gizle":"Pred Göster"}
                </button>
              )}
              <button onClick={submit}
                style={{padding:"6px 16px",borderRadius:6,border:"none",background:C.accent,color:"#fff",fontSize:12,fontWeight:600,cursor:"pointer"}}>
                {imgIdx+1>=sessionImgs.length?"Oturumu Bitir":"Sonraki →"}
              </button>
            </div>

            <AnnotationCanvas
              imageData={imageData} boxes={boxes} selectedIdx={selIdx}
              onBoxesChange={setBoxes} onSelectIdx={setSelIdx}
              showGt={showGt} showPred={mode==="ai"&&showPred}
            />

            {/* Araç çubuğu */}
            <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginTop:8,flexWrap:"wrap",gap:6}}>
              <div style={{display:"flex",gap:5,alignItems:"center",flexWrap:"wrap"}}>
                {boxes.map((_,i)=>(
                  <button key={i} onClick={()=>setSelIdx(selIdx===i?null:i)}
                    style={{padding:"3px 9px",borderRadius:5,border:`1px solid ${selIdx===i?C.physician:C.border}`,background:selIdx===i?C.physicianSoft:"transparent",color:selIdx===i?C.physician:C.textMuted,fontSize:11,cursor:"pointer",fontFamily:"monospace"}}>
                    #{i+1}
                  </button>
                ))}
                {!boxes.length&&<span style={{fontSize:11,color:C.textDim}}>Bbox yok</span>}
              </div>
              <div style={{display:"flex",gap:6}}>
                {selIdx!==null&&(
                  <button onClick={()=>{setBoxes(b=>b.filter((_,i)=>i!==selIdx));setSelIdx(null);}}
                    style={{padding:"4px 10px",borderRadius:6,border:`1px solid ${C.danger}44`,background:C.dangerSoft,color:C.danger,fontSize:11,cursor:"pointer"}}>
                    #{selIdx+1} Sil
                  </button>
                )}
                {boxes.length>0&&(
                  <button onClick={()=>{setBoxes([]);setSelIdx(null);}}
                    style={{padding:"4px 10px",borderRadius:6,border:`1px solid ${C.border}`,background:"transparent",color:C.textMuted,fontSize:11,cursor:"pointer"}}>
                    Tümünü Sil
                  </button>
                )}

              </div>
            </div>

            <div style={{fontSize:10,color:C.textDim,marginTop:5}}>
              Sürükle → yeni bbox · Tıkla → seç / taşı · Köşe handle → boyutlandır · Delete → sil
            </div>

            <div style={{display:"flex",alignItems:"center",justifyContent:"flex-start",marginTop:14}}>
              <div style={{fontSize:12,color:C.textMuted}}>
                {boxes.length===0
                  ?<span>Patoloji yoksa boş bırakabilirsiniz</span>
                  :<span style={{color:C.success}}>✓ {boxes.length} bbox — ilerleyebilirsiniz</span>}
              </div>
            </div>
          </div>
        )}

        {phase==="report"&&<ReportView results={results} mode={mode} onReset={reset}/>}
      </div>
    </div>
  );
}
