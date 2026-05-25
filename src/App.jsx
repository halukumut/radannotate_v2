import { useState, useRef, useEffect, useCallback } from "react";

// ── Image Cache Hook ──────────────────────────────────────────────────────
// Pre-loads images to reduce lag when switching between images
function useImageCache(sessionImgs, currentIdx) {
  const cacheRef = useRef(new Map());
  const [preloadedUrls, setPreloadedUrls] = useState(new Set());

  // Pre-cache next image and session starter images
  useEffect(() => {
    if (!sessionImgs.length) return;

    const toLoad = new Set();

    // Add next image
    if (currentIdx + 1 < sessionImgs.length) {
      toLoad.add(sessionImgs[currentIdx + 1].url);
    }

    // Add first two images of session (if not already loaded)
    if (currentIdx === 0 && sessionImgs.length > 0) {
      toLoad.add(sessionImgs[0].url);
      if (sessionImgs.length > 1) toLoad.add(sessionImgs[1].url);
    }

    toLoad.forEach((url) => {
      if (preloadedUrls.has(url)) return; // Skip if already loaded
      const img = new Image();
      img.onload = () => {
        cacheRef.current.set(url, img);
        setPreloadedUrls((prev) => new Set([...prev, url]));
      };
      img.src = url;
    });

    return () => {
      // Clean up cache when moving to next image
      const urlsToKeep = new Set([
        sessionImgs[currentIdx]?.url,
        sessionImgs[currentIdx + 1]?.url,
      ]);
      cacheRef.current.forEach((_, url) => {
        if (!urlsToKeep.has(url)) {
          cacheRef.current.delete(url);
        }
      });
    };
  }, [sessionImgs, currentIdx, preloadedUrls]);

  return cacheRef.current;
}

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

// Calculate aggregated session statistics from results
function buildSessionStats(results) {
  const times = results.map(r => r.time);
  const ious = results.map(r => r.iou).filter(v => v != null);
  const correct = results.filter(r => r.labelCorrect).length;
  const total = times.reduce((a, b) => a + b, 0);
  const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

  return {
    imageCount: results.length,
    correctCount: correct,
    accuracyRate: +(correct / results.length).toFixed(4),
    totalTimeMs: total,
    avgTimeMs: times.length ? +avg(times).toFixed(1) : null,
    minTimeMs: times.length ? Math.min(...times) : null,
    maxTimeMs: times.length ? Math.max(...times) : null,
    avgIou: ious.length ? +avg(ious).toFixed(4) : null,
    minIou: ious.length ? +Math.min(...ious).toFixed(4) : null,
    maxIou: ious.length ? +Math.max(...ious).toFixed(4) : null,
    annotatedImages: results.filter(r => r.boxes?.length > 0).length,
    emptyImages: results.filter(r => !r.boxes?.length).length,
  };
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

function getHandles(b, ox, oy, W, H) {
  const bx = ox + b.x * W, by = oy + b.y * H, bw = b.w * W, bh = b.h * H;
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
  for(const h of getHandles(b,0,0,W,H))
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
function AnnotationCanvas({imageData,boxes,selectedIdx,onBoxesChange,onSelectIdx,showGt,showPred,aiMode}){
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
  // Keep pred boxes accessible in event handlers without stale closure
  const predBoxesRef = useRef(imageData?.pred??[]);
  useEffect(()=>{predBoxesRef.current=imageData?.pred??[];},[imageData?.pred]);
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

    // Hekim bbox'ları (AI oturumda: source="ai" → mor kesik, source="edited"/"physician" → turuncu düz)
    boxesRef.current.forEach((b,i)=>{
      const sel=i===selRef.current;
      const isAiSource = b.source === "ai";
      const color = isAiSource ? C.ai : C.physician;
      const dash  = isAiSource ? [6,3] : null;
      let label;
      if(boxesRef.current.length>1) label=isAiSource?`AI #${i+1}`:`#${i+1}`;
      else label=isAiSource?"AI (onaylandı)":"Hekim";
      drawBox(b, color, dash, label, sel?1:0.82);
      if(sel){
        getHandles(b,ox,oy,displayW,displayH).forEach(hd=>{
          ctx.fillStyle=color; ctx.strokeStyle="#060810"; ctx.lineWidth=1.5;
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
    // Also show pointer over pred boxes (clickable to import)
    if(predBoxesRef.current?.length){
      for(const pb of predBoxesRef.current){
        const r=yoloToRect(pb);
        if(hitBox(r,px-ox,py-oy,displayW,displayH)) return "pointer";
      }
    }
    return "crosshair";
  };

  const onDown=(e)=>{
    if(!imageData) return;
    e.preventDefault();
    const {px,py,nx,ny}=getPos(e);
    const {ox, oy, w: displayW, h: displayH} = displayDimsRef.current;
    const bxs=boxesRef.current; const sel=selRef.current;
    const inter=interRef.current;

    if(sel!==null){
      const hd=hitHandle(bxs[sel],px-ox,py-oy,displayW,displayH);
      if(hd){inter.mode="resize";inter.startPx=px;inter.startPy=py;inter.origBox={...bxs[sel]};inter.handleId=hd.id;return;}
      if(hitBox(bxs[sel],px-ox,py-oy,displayW,displayH)){inter.mode="move";inter.startPx=px;inter.startPy=py;inter.origBox={...bxs[sel]};return;}
    }
    // Hit test existing editable boxes
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
      onBoxesChange(boxesRef.current.map((b,i)=>i===sel
        ?{...b,source:"edited",x:Math.max(0,Math.min(1-ob.w,ob.x+dx)),y:Math.max(0,Math.min(1-ob.h,ob.y+dy))}
        :b));
    }
    else if(inter.mode==="resize"&&inter.origBox){
      const dx=(px-inter.startPx)/displayW, dy=(py-inter.startPy)/displayH;
      const sel=selRef.current; if(sel===null) return;
      onBoxesChange(boxesRef.current.map((b,i)=>i===sel
        ?{...applyResize(inter.origBox,inter.handleId,dx,dy),source:"edited"}
        :b));
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
        const nb=[...boxesRef.current,{id:Date.now(),x,y,w:bw,h:bh,source:"physician"}];
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

// ── Physician Name Input Modal ────────────────────────────────────────────
function NameInputModal({onStart}){
  const [name,setName]=useState("");
  const [error,setError]=useState("");

  const handleSubmit=()=>{
    const trimmed=name.trim();
    if(!trimmed){setError("Lütfen adınızı girin");return;}
    if(trimmed.length<2){setError("Ad en az 2 karakter olmalı");return;}
    onStart(trimmed);
  };

  return (
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.7)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:100}}>
      <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:28,maxWidth:360,boxShadow:"0 20px 60px rgba(0,0,0,0.5)"}}>
        <div style={{fontSize:18,fontWeight:700,color:C.text,marginBottom:8}}>Adınızı Girin</div>
        <div style={{fontSize:13,color:C.textMuted,marginBottom:16}}>Bu oturum için doktor adınızı yazın</div>
        <input
          type="text"
          value={name}
          onChange={(e)=>{setName(e.target.value);setError("");}}
          placeholder="Örn: Dr. Ahmet"
          autoFocus
          onKeyPress={(e)=>e.key==="Enter"&&handleSubmit()}
          style={{width:"100%",padding:10,borderRadius:6,border:`1px solid ${C.border}`,background:C.bg,color:C.text,fontSize:13,marginBottom:8,outline:"none",boxSizing:"border-box"}}
        />
        {error&&<div style={{fontSize:12,color:C.danger,marginBottom:12}}>{error}</div>}
        <button onClick={handleSubmit}
          style={{width:"100%",padding:10,background:C.accent,color:"#fff",border:"none",borderRadius:6,fontSize:13,fontWeight:600,cursor:"pointer"}}>
          Başla
        </button>
      </div>
    </div>
  );
}

// ── Session Instruction Modal ────────────────────────────────────────────
function SessionInstructionModal({mode,onAcknowledge}){
  const instructions={
    solo:{
      title:"Oturum A — Solo Anotasyon",
      description:"AI yardımı olmadan bağımsız olarak tıbbi görüntüleri etiketleyin.",
      steps:[
        "📌 Kutucuk çizmek için görsele tıklayıp sürükleyin",
        "✏️ Kutucukları düzenlemek için köşe ve kenarları ayarlayın",
        "🗑️ Kutucukları silmek için seçtikten sonra Delete tuşuna basın",
        "⚫ Patoloji yoksa boş bırakabilirsiniz",
      ],
      tips:"Görseli daha iyi görmek için sağ üstteki tam ekran simgesini kullanabilirsiniz."
    },
    ai:{
      title:"Oturum B — AI Destekli Anotasyon",
      description:"Model tarafından önerilen kutucukları gözden geçirin ve düzenleyin.",
      steps:[
        "🤖 Model önerisi AI kutucukları otomatik gösterilir",
        "✅ Öneriyle aynı kalması için hiçbir şey yapmayın",
        "📌 Kutucuk çizmek için görsele tıklayıp sürükleyin",
        "✏️ Kutucukları düzenlemek için köşe ve kenarları ayarlayın",
        "🗑️ Kutucukları silmek için seçtikten sonra Delete tuşuna basın",
        "⚠️ Hiçbir işlem yapmazsanız, AI önerisi kabul edilir"
      ],
      tips:"Görseli daha iyi görmek için sağ üstteki tam ekran simgesini kullanabilirsiniz."
    }
  };

  const inst=instructions[mode]||instructions.solo;

  return (
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.85)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:200}}>
      <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:16,padding:32,maxWidth:520,boxShadow:"0 20px 80px rgba(0,0,0,0.6)",maxHeight:"90vh",overflow:"auto"}}>
        <div style={{fontSize:11,color:mode==="ai"?C.ai:C.physician,textTransform:"uppercase",letterSpacing:"0.08em",fontWeight:700,marginBottom:8}}>
          {mode==="ai"?"🤖 AI Destekli":"👁 Solo"} Mod
        </div>
        <div style={{fontSize:22,fontWeight:800,color:C.text,marginBottom:8}}>{inst.title}</div>
        <div style={{fontSize:14,color:C.textMuted,lineHeight:1.6,marginBottom:20}}>{inst.description}</div>

        <div style={{background:C.bg,border:`1px solid ${C.border}`,borderRadius:10,padding:16,marginBottom:20}}>
          <div style={{fontSize:12,fontWeight:700,color:C.text,marginBottom:12,textTransform:"uppercase",letterSpacing:"0.04em"}}>Nasıl Kullanacağım?</div>
          {inst.steps.map((step,i)=>(
            <div key={i} style={{fontSize:13,color:C.text,lineHeight:1.7,marginBottom:8,display:"flex",gap:10}}>
              <span style={{minWidth:24,fontSize:12}}>{step}</span>
            </div>
          ))}
        </div>

        <div style={{background:C.accentSoft,border:`1px solid ${C.accent}44`,borderRadius:10,padding:12,marginBottom:20}}>
          <div style={{fontSize:12,color:C.accent,fontWeight:600}}>💡 İpucu</div>
          <div style={{fontSize:12,color:C.accent,lineHeight:1.6,marginTop:6}}>{inst.tips}</div>
        </div>

        <button onClick={onAcknowledge}
          style={{width:"100%",padding:12,background:C.accent,color:"#fff",border:"none",borderRadius:8,fontSize:14,fontWeight:700,cursor:"pointer",transition:"opacity 0.2s"}}>
          Anladım, Başlayalım
        </button>
      </div>
    </div>
  );
}

// ── Fullscreen Image Viewer (Zoom + Pan) with full bbox editing ──────────
function FullscreenViewer({imageData,boxes,selectedIdx,onBoxesChange,onSelectIdx,showGt,showPred,onShowPredToggle,aiMode,mode,imgIdx,sessionLength,onSubmit,onClose}){
  const [zoom,setZoom]=useState(1);
  const [pan,setPan]=useState({x:0,y:0});
  const canvasRef=useRef(null);
  const imgRef=useRef(null);
  const imgDimsRef=useRef({w:1,h:1});
  const boxesRef=useRef(boxes);
  const selRef=useRef(selectedIdx);
  const predBoxesRef=useRef(imageData?.pred??[]);
  const panStartRef=useRef(null);
  const interRef=useRef({mode:"idle"});
  const zoomRef=useRef(zoom);
  const HR=6;

  useEffect(()=>{zoomRef.current=zoom;},[zoom]);
  useEffect(()=>{predBoxesRef.current=imageData?.pred??[];},[imageData?.pred]);
  useEffect(()=>{boxesRef.current=boxes;},[boxes]);
  useEffect(()=>{selRef.current=selectedIdx;},[selectedIdx]);

  // Load image
  useEffect(()=>{
    imgRef.current=null;
    if(!imageData?.url) return;
    const img=new Image();
    img.onload=()=>{imgRef.current=img;draw();};
    img.onerror=()=>{imgRef.current=null;draw();};
    img.src=imageData.url;
  },[imageData?.url]);

  // Store original image dimensions
  useEffect(()=>{
    if(!imgRef.current) return;
    const iw=imgRef.current.naturalWidth||imgRef.current.width;
    const ih=imgRef.current.naturalHeight||imgRef.current.height;
    if(iw>0&&ih>0) imgDimsRef.current={w:iw,h:ih};
  },[imageData?.url]);

  // Keyboard shortcuts
  useEffect(()=>{
    const onKey=(e)=>{
      if(e.key==="Escape") onClose();
      else if(e.key==="+" || e.key==="=") setZoom(z=>Math.min(3,z+0.2));
      else if(e.key==="-") setZoom(z=>Math.max(1,z-0.2));
      else if(e.key==="0") setZoom(1);
      else if((e.key==="Delete"||e.key==="Backspace")&&selRef.current!==null){
        onBoxesChange(boxesRef.current.filter((_,i)=>i!==selRef.current));
        onSelectIdx(null);
      }
    };
    window.addEventListener("keydown",onKey);
    return ()=>window.removeEventListener("keydown",onKey);
  },[onBoxesChange,onSelectIdx,onClose]);

  const getHandles=(b,ox,oy,W,H)=>{
    const bx=ox+b.x*W,by=oy+b.y*H,bw=b.w*W,bh=b.h*H;
    return [
      {id:"tl",cx:bx,cy:by,cursor:"nwse-resize"},
      {id:"tm",cx:bx+bw/2,cy:by,cursor:"ns-resize"},
      {id:"tr",cx:bx+bw,cy:by,cursor:"nesw-resize"},
      {id:"ml",cx:bx,cy:by+bh/2,cursor:"ew-resize"},
      {id:"mr",cx:bx+bw,cy:by+bh/2,cursor:"ew-resize"},
      {id:"bl",cx:bx,cy:by+bh,cursor:"nesw-resize"},
      {id:"bm",cx:bx+bw/2,cy:by+bh,cursor:"ns-resize"},
      {id:"br",cx:bx+bw,cy:by+bh,cursor:"nwse-resize"},
    ];
  };

  const applyResize=(b,hid,dx,dy)=>{
    let {x,y,w,h}=b;
    if(hid.includes("l")){x+=dx;w-=dx;}if(hid.includes("r")){w+=dx;}
    if(hid.includes("t")){y+=dy;h-=dy;}if(hid.includes("b")){h+=dy;}
    if(w<0.01){if(hid.includes("l"))x=b.x+b.w-0.01;w=0.01;}
    if(h<0.01){if(hid.includes("t"))y=b.y+b.h-0.01;h=0.01;}
    x=Math.max(0,x);y=Math.max(0,y);
    w=Math.min(1-x,w);h=Math.min(1-y,h);
    return {...b,x,y,w,h};
  };

  const hitHandle=(b,px,py,W,H)=>{
    for(const h of getHandles(b,0,0,W,H))
      if(Math.abs(px-h.cx)<=HR+2&&Math.abs(py-h.cy)<=HR+2) return h;
    return null;
  };

  const hitBox=(b,px,py,W,H)=>{
    return px>=b.x*W&&px<=(b.x+b.w)*W&&py>=b.y*H&&py<=(b.y+b.h)*H;
  };

  const getPos=(e,ox,oy,displayW,displayH)=>{
    const canvas=canvasRef.current;
    const rect=canvas.getBoundingClientRect();
    const px=e.clientX-rect.left,py=e.clientY-rect.top;
    const nx=(px-ox)/displayW,ny=(py-oy)/displayH;
    return {px,py,nx,ny};
  };

  const draw=useCallback(()=>{
    const canvas=canvasRef.current;
    if(!canvas) return;
    const ctx=canvas.getContext("2d");
    const W=window.innerWidth-32;
    const H=window.innerHeight-120;
    canvas.width=W;
    canvas.height=H;

    ctx.fillStyle=C.bg;ctx.fillRect(0,0,W,H);

    // Grid
    ctx.strokeStyle=C.border;ctx.lineWidth=0.5;ctx.globalAlpha=0.2;
    for(let x=0;x<W;x+=32){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=0;y<H;y+=32){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    ctx.globalAlpha=1;

    if(imgRef.current){
      const iw=imgDimsRef.current.w;
      const ih=imgDimsRef.current.h;
      const z=zoomRef.current;
      const scale=Math.min(W/iw,H/ih)*z;
      const displayW=Math.round(iw*scale);
      const displayH=Math.round(ih*scale);
      const baseX=(W-displayW)/2;
      const baseY=(H-displayH)/2;
      const ox=baseX+pan.x;
      const oy=baseY+pan.y;
      ctx.drawImage(imgRef.current,ox,oy,displayW,displayH);

      const drawBox=(rect,color,dash,label,alpha=1)=>{
        const px=ox+rect.x*displayW,py=oy+rect.y*displayH,pw=rect.w*displayW,ph=rect.h*displayH;
        ctx.globalAlpha=alpha;
        ctx.strokeStyle=color;ctx.lineWidth=2;
        if(dash)ctx.setLineDash(dash);else ctx.setLineDash([]);
        ctx.strokeRect(px,py,pw,ph);
        ctx.fillStyle=color+"20";ctx.fillRect(px,py,pw,ph);
        ctx.setLineDash([]);
        if(label){
          const tw=ctx.measureText(label).width+8;
          ctx.fillStyle=color+"ee";
          ctx.fillRect(px,py-16,tw,16);
          ctx.fillStyle="#fff";ctx.font="bold 10px monospace";
          ctx.fillText(label,px+4,py-4);
        }
        ctx.globalAlpha=1;
      };

      if(showGt&&imageData?.gt){
        imageData.gt.forEach((gb,i)=>{
          drawBox({x:gb.cx-gb.w/2,y:gb.cy-gb.h/2,w:gb.w,h:gb.h},C.gt,[5,4],imageData.gt.length>1?`GT #${i+1}`:"GT");
        });
      }

      if(showPred&&imageData?.pred){
        imageData.pred.forEach((pb,i)=>{
          const conf=pb.conf??1;
          const col=conf>=0.80?C.success:conf>=0.55?C.warning:C.danger;
          drawBox({x:pb.cx-pb.w/2,y:pb.cy-pb.h/2,w:pb.w,h:pb.h},col,[8,4],`AI ${Math.round(conf*100)}%${imageData.pred.length>1?` #${i+1}`:""}`,0.9);
        });
      }

      boxesRef.current.forEach((b,i)=>{
        const sel=i===selRef.current;
        const isAiSource=b.source==="ai";
        const color=isAiSource?C.ai:C.physician;
        const dash=isAiSource?[6,3]:null;
        let label;
        if(boxesRef.current.length>1)label=isAiSource?`AI #${i+1}`:`#${i+1}`;
        else label=isAiSource?"AI":"Hekim";
        drawBox(b,color,dash,label,sel?1:0.82);
        if(sel){
          getHandles(b,ox,oy,displayW,displayH).forEach(hd=>{
            ctx.fillStyle=color;ctx.strokeStyle="#060810";ctx.lineWidth=1.5;
            ctx.beginPath();ctx.rect(hd.cx-HR,hd.cy-HR,HR*2,HR*2);ctx.fill();ctx.stroke();
          });
        }
      });

      // Draw preview
      const inter=interRef.current;
      if(inter.mode==="drawing"&&inter.drawStart&&inter.drawCurrent){
        const ds=inter.drawStart,dc=inter.drawCurrent;
        const x=Math.min(ds.x,dc.x),y=Math.min(ds.y,dc.y);
        const bw=Math.abs(dc.x-ds.x),bh=Math.abs(dc.y-ds.y);
        if(bw>0.005&&bh>0.005)drawBox({x,y,w:bw,h:bh},C.physician,[5,3],null);
      }
    }else{
      ctx.fillStyle=C.textDim;ctx.font="16px monospace";ctx.textAlign="center";
      ctx.fillText(imageData?.file??"Görsel bekleniyor…",W/2,H/2);
    }
  },[imageData,showGt,showPred]);

  useEffect(()=>{draw();},[draw,boxes,selectedIdx,zoom,pan]);

  const onCanvasMouseDown=(e)=>{
    if(!imageData) return;
    e.preventDefault();
    canvasRef.current?.focus();
    
    const canvas=canvasRef.current;
    const rect=canvas.getBoundingClientRect();
    const W=rect.width,H=rect.height;
    const iw=imgDimsRef.current.w,ih=imgDimsRef.current.h;
    const z=zoomRef.current;
    const scale=Math.min(W/iw,H/ih)*z;
    const displayW=Math.round(iw*scale);
    const displayH=Math.round(ih*scale);
    const baseX=(W-displayW)/2;
    const baseY=(H-displayH)/2;
    const ox=baseX+pan.x;
    const oy=baseY+pan.y;

    if(zoom>1&&e.button===0){
      panStartRef.current={x:e.clientX,y:e.clientY,startPan:{...pan}};
      return;
    }

    const px=e.clientX-rect.left,py=e.clientY-rect.top;
    const nx=(px-ox)/displayW,ny=(py-oy)/displayH;
    const inter=interRef.current;
    const bxs=boxesRef.current;
    const sel=selRef.current;

    if(sel!==null){
      const hd=hitHandle(bxs[sel],px-ox,py-oy,displayW,displayH);
      if(hd){inter.mode="resize";inter.startPx=px;inter.startPy=py;inter.origBox={...bxs[sel]};inter.handleId=hd.id;return;}
      if(hitBox(bxs[sel],px-ox,py-oy,displayW,displayH)){inter.mode="move";inter.startPx=px;inter.startPy=py;inter.origBox={...bxs[sel]};return;}
    }

    for(let i=bxs.length-1;i>=0;i--){
      if(hitBox(bxs[i],px-ox,py-oy,displayW,displayH)){
        onSelectIdx(i);selRef.current=i;
        inter.mode="move";inter.startPx=px;inter.startPy=py;inter.origBox={...bxs[i]};return;
      }
    }

    onSelectIdx(null);selRef.current=null;
    inter.mode="drawing";inter.drawStart={x:nx,y:ny};inter.drawCurrent={x:nx,y:ny};
  };

  const onCanvasMouseMove=(e)=>{
    const canvas=canvasRef.current;
    const rect=canvas.getBoundingClientRect();
    const W=rect.width,H=rect.height;
    const iw=imgDimsRef.current.w,ih=imgDimsRef.current.h;
    const z=zoomRef.current;
    const scale=Math.min(W/iw,H/ih)*z;
    const displayW=Math.round(iw*scale);
    const displayH=Math.round(ih*scale);
    const baseX=(W-displayW)/2;
    const baseY=(H-displayH)/2;
    const ox=baseX+pan.x;
    const oy=baseY+pan.y;

    const px=e.clientX-rect.left,py=e.clientY-rect.top;
    const nx=(px-ox)/displayW,ny=(py-oy)/displayH;
    const inter=interRef.current;

    if(panStartRef.current){
      const dx=e.clientX-panStartRef.current.x;
      const dy=e.clientY-panStartRef.current.y;
      setPan({x:panStartRef.current.startPan.x+dx,y:panStartRef.current.startPan.y+dy});
      return;
    }

    if(inter.mode==="drawing"){inter.drawCurrent={x:nx,y:ny};draw();}
    else if(inter.mode==="move"&&inter.origBox){
      const dx=(px-inter.startPx)/displayW,dy=(py-inter.startPy)/displayH;
      const sel=selRef.current;if(sel===null)return;
      const ob=inter.origBox;
      onBoxesChange(boxesRef.current.map((b,i)=>i===sel?{...b,source:"edited",x:Math.max(0,Math.min(1-ob.w,ob.x+dx)),y:Math.max(0,Math.min(1-ob.h,ob.y+dy))}:b));
    }
    else if(inter.mode==="resize"&&inter.origBox){
      const dx=(px-inter.startPx)/displayW,dy=(py-inter.startPy)/displayH;
      const sel=selRef.current;if(sel===null)return;
      onBoxesChange(boxesRef.current.map((b,i)=>i===sel?{...applyResize(inter.origBox,inter.handleId,dx,dy),source:"edited"}:b));
    }
  };

  const onCanvasMouseUp=(e)=>{
    const canvas=canvasRef.current;
    const rect=canvas.getBoundingClientRect();
    const W=rect.width,H=rect.height;
    const iw=imgDimsRef.current.w,ih=imgDimsRef.current.h;
    const z=zoomRef.current;
    const scale=Math.min(W/iw,H/ih)*z;
    const displayW=Math.round(iw*scale);
    const displayH=Math.round(ih*scale);
    const baseX=(W-displayW)/2;
    const baseY=(H-displayH)/2;
    const ox=baseX+pan.x;
    const oy=baseY+pan.y;

    const px=e.clientX-rect.left,py=e.clientY-rect.top;
    const nx=(px-ox)/displayW,ny=(py-oy)/displayH;
    const inter=interRef.current;

    if(inter.mode==="drawing"&&inter.drawStart){
      const ds=inter.drawStart;
      const x=Math.min(ds.x,nx),y=Math.min(ds.y,ny);
      const bw=Math.abs(nx-ds.x),bh=Math.abs(ny-ds.y);
      if(bw>0.02&&bh>0.02){
        const nb=[...boxesRef.current,{id:Date.now(),x,y,w:bw,h:bh,source:"physician"}];
        onBoxesChange(nb);onSelectIdx(nb.length-1);
      }
    }

    inter.mode="idle";inter.drawStart=null;inter.drawCurrent=null;inter.origBox=null;
    panStartRef.current=null;
    draw();
  };

  const onWheel=(e)=>{
    if(!e.ctrlKey) return;
    e.preventDefault();
    const delta=e.deltaY>0?-0.15:0.15;
    setZoom(z=>Math.max(1,Math.min(3,z+delta)));
  };

  return (
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.95)",display:"flex",flexDirection:"column",zIndex:150}}>
      {/* Top toolbar */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"8px 16px",background:C.surface,borderBottom:`1px solid ${C.border}`}}>
        <div style={{fontSize:13,color:C.textMuted}}>Tam Ekran · {imageData?.file||"Görsel"} · {imgIdx+1}/{sessionLength}</div>
        <div style={{display:"flex",gap:10,alignItems:"center"}}>
          <span style={{fontSize:12,color:C.textMuted,fontFamily:"monospace"}}>{Math.round(zoom*100)}%</span>
          <button onClick={()=>setZoom(1)} title="Ctrl+0" style={{padding:"4px 8px",borderRadius:4,border:`1px solid ${C.border}`,background:"transparent",color:C.text,fontSize:11,cursor:"pointer"}}>
            Sıfırla
          </button>
          <button onClick={()=>setZoom(z=>Math.max(1,z-0.1))} title="Ctrl+-" style={{padding:"4px 8px",borderRadius:4,border:`1px solid ${C.border}`,background:"transparent",color:C.text,fontSize:11,cursor:"pointer"}}>
            −
          </button>
          <button onClick={()=>setZoom(z=>Math.min(3,z+0.1))} title="Ctrl++" style={{padding:"4px 8px",borderRadius:4,border:`1px solid ${C.border}`,background:"transparent",color:C.text,fontSize:11,cursor:"pointer"}}>
            +
          </button>
          <button onClick={onClose} style={{padding:"4px 10px",borderRadius:4,border:`1px solid ${C.border}`,background:"transparent",color:C.text,fontSize:11,cursor:"pointer"}}>
            ✕ Çık
          </button>
        </div>
      </div>

      {/* Canvas with overlay panels */}
      <div style={{flex:1,position:"relative",overflow:"hidden"}} onMouseDown={onCanvasMouseDown} onMouseMove={onCanvasMouseMove} onMouseUp={onCanvasMouseUp} onMouseLeave={onCanvasMouseUp} onWheel={onWheel}>
        <canvas ref={canvasRef} style={{cursor:"crosshair",display:"block",width:"100%",height:"100%"}}/>
        
        {/* AI Predictions Panel (top left) - fades when zoomed */}
        {aiMode&&imageData?.pred?.length>0&&(
          <div style={{position:"absolute",top:12,left:12,background:C.aiSoft,border:`1px solid ${C.ai}33`,borderRadius:8,padding:"10px 12px",fontSize:11,maxWidth:200,transition:"opacity 0.3s",opacity:zoom>1.5?0.3:1}}>
            <div style={{fontSize:10,color:C.ai,textTransform:"uppercase",letterSpacing:"0.06em",marginBottom:6,fontWeight:700}}>AI Tahminleri</div>
            {imageData.pred.map((pb,i)=>{
              const conf=pb.conf??1;
              const col=conf>=0.80?C.success:conf>=0.55?C.warning:C.danger;
              const pct=Math.round(conf*100);
              return (
                <div key={i} style={{display:"flex",alignItems:"center",gap:6,marginBottom:4}}>
                  <span style={{fontSize:10,color:C.textDim,fontFamily:"monospace",minWidth:16}}>#{i+1}</span>
                  <div style={{flex:1,background:C.border,borderRadius:99,height:4,overflow:"hidden"}}>
                    <div style={{width:`${pct}%`,height:"100%",background:col,borderRadius:99}}/>
                  </div>
                  <span style={{fontSize:9,fontWeight:600,color:col,fontFamily:"monospace"}}>%{pct}</span>
                </div>
              );
            })}
          </div>
        )}

        {zoom>1&&<div style={{position:"absolute",bottom:16,left:16,fontSize:11,color:C.textMuted,background:C.surface,border:`1px solid ${C.border}`,borderRadius:6,padding:"8px 12px"}}>
          Ctrl+Scroll: Zoom · Sürükle: Pan · Esc: Çık
        </div>}
      </div>

      {/* Bottom toolbar */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"12px 16px",background:C.surface,borderTop:`1px solid ${C.border}`,gap:10}}>
        <div style={{display:"flex",gap:8,alignItems:"center",flex:1}}>
          {aiMode&&(
            <button onClick={onShowPredToggle}
              style={{padding:"6px 14px",borderRadius:6,border:`1px solid ${C.ai}44`,background:showPred?C.aiSoft:"transparent",color:C.ai,fontSize:12,cursor:"pointer",fontWeight:500}}>
              {showPred?"Pred Gizle":"Pred Göster"}
            </button>
          )}
          <div style={{flex:1,maxWidth:300}}>
            <div style={{background:C.border,borderRadius:99,height:6,overflow:"hidden"}}>
              <div style={{width:`${((imgIdx+1)/sessionLength)*100}%`,height:"100%",background:C.accent,borderRadius:99,transition:"width 0.4s ease"}}/>
            </div>
          </div>
        </div>
        <button onClick={onSubmit}
          style={{padding:"8px 18px",borderRadius:6,border:"none",background:C.accent,color:"#fff",fontSize:12,fontWeight:600,cursor:"pointer"}}>
          {imgIdx+1>=sessionLength?"Oturumu Bitir":"Sonraki →"}
        </button>
      </div>
    </div>
  );
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

// ── Comparison Canvas (보기 전용, bbox 표시) ─────────────────────────────────
function ComparisonCanvas({imageData,physicianBoxes,gtYoloBoxes,predBoxes,showGt,showPhysician,showPred}){
  const ASPECT_RATIO = 700 / 900;
  const [canvasSize, setCanvasSize] = useState({w:700,h:900});
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const imgDimsRef = useRef({w:1,h:1});
  const displayDimsRef = useRef({ox:0, oy:0, w:700, h:467});
  const canvasSizeRef = useRef(canvasSize);
  useEffect(()=>{canvasSizeRef.current=canvasSize;},[canvasSize]);

  useEffect(()=>{
    const ro = new ResizeObserver(([e])=>{
      const w=e.contentRect.width;
      const maxW=Math.min(w,700);
      setCanvasSize({w:maxW, h:Math.round(maxW/ASPECT_RATIO)});
    });
    if(canvasRef.current?.parentElement) ro.observe(canvasRef.current.parentElement);
    return ()=>ro.disconnect();
  },[]);

  useEffect(()=>{
    if(!imgRef.current) return;
    const iw = imgRef.current.naturalWidth || imgRef.current.width;
    const ih = imgRef.current.naturalHeight || imgRef.current.height;
    if(iw>0 && ih>0) {
      imgDimsRef.current = {w:iw, h:ih};
      const {w: CANVAS_W, h: CANVAS_H} = canvasSizeRef.current;
      const scale = Math.min(CANVAS_W / iw, CANVAS_H / ih);
      const displayW = Math.round(iw * scale);
      const displayH = Math.round(ih * scale);
      const ox = Math.round((CANVAS_W - displayW) / 2);
      const oy = Math.round((CANVAS_H - displayH) / 2);
      displayDimsRef.current = {ox, oy, w: displayW, h: displayH};
    }
  },[imageData?.url]);

  useEffect(()=>{
    if(!imageData?.url) return;
    const img=new Image();
    img.onload=()=>{imgRef.current=img; draw();};
    img.onerror=()=>{imgRef.current=null; draw();};
    img.src=imageData.url;
  },[imageData?.url]);

  const draw = useCallback(()=>{
    const canvas=canvasRef.current; if(!canvas) return;
    const ctx=canvas.getContext("2d");
    const {w: CANVAS_W, h: CANVAS_H} = canvasSizeRef.current;
    canvas.width=CANVAS_W; canvas.height=CANVAS_H;

    ctx.fillStyle="#060810"; ctx.fillRect(0,0,CANVAS_W,CANVAS_H);

    const {ox, oy, w: displayW, h: displayH} = displayDimsRef.current;
    if(imgRef.current){
      ctx.drawImage(imgRef.current, ox, oy, displayW, displayH);
    } else {
      ctx.strokeStyle="#1a2030"; ctx.lineWidth=1;
      for(let x=0;x<CANVAS_W;x+=32){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,CANVAS_H);ctx.stroke();}
      for(let y=0;y<CANVAS_H;y+=32){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(CANVAS_W,y);ctx.stroke();}
      ctx.fillStyle=C.textDim; ctx.font="12px monospace"; ctx.textAlign="center";
      ctx.fillText(imageData?.file??"Görsel bekleniyor…",CANVAS_W/2,CANVAS_H/2);
      ctx.textAlign="left";
    }

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

    if(showGt && gtYoloBoxes?.length){
      gtYoloBoxes.forEach((gb,i)=>{
        drawBox(yoloToRect(gb), C.gt, [5,4],
          gtYoloBoxes.length>1?`GT #${i+1}`:"GT");
      });
    }

    if(showPred && predBoxes?.length){
      predBoxes.forEach((pb,i)=>{
        const col=confColor(pb.conf??1);
        drawBox(yoloToRect(pb), col, [8,4],
          `AI ${Math.round((pb.conf??1)*100)}%${predBoxes.length>1?` #${i+1}`:""}`,
          0.9);
      });
    }

    if(showPhysician && physicianBoxes?.length){
      physicianBoxes.forEach((b,i)=>{
        drawBox(b, C.physician, null,
          physicianBoxes.length>1?`#${i+1}`:"Hekim",
          1);
      });
    }
  },[imageData,gtYoloBoxes,predBoxes,physicianBoxes,showGt,showPhysician,showPred,canvasSize]);

  useEffect(()=>{draw();},[draw]);

  const {w,h}=canvasSize;
  return <canvas ref={canvasRef} width={w} height={h}
    style={{width:w,height:h,borderRadius:8,border:`1px solid ${C.border}`,display:"block",userSelect:"none"}}
  />;
}

// ── Comparison View (Results detaylı görünüş) ─────────────────────────────
function ComparisonView({result,sessionImgs,onBack,onFeedbackSubmit}){
  const imageData=sessionImgs.find(img=>img.id===result.imageId);
  const [feedback,setFeedback]=useState(result.feedback||"");
  const [submitted,setSubmitted]=useState(!!result.feedback);
  const [showGt,setShowGt]=useState(true);
  const [showPhysician,setShowPhysician]=useState(true);
  const [showPred,setShowPred]=useState(imageData?.pred?.length>0);

  const handleSubmit=()=>{
    onFeedbackSubmit(result.imageId,feedback);
    setSubmitted(true);
  };

  return (
    <div style={{paddingTop:20}}>
      <button onClick={onBack} style={{marginBottom:20,padding:"8px 16px",borderRadius:6,border:`1px solid ${C.border}`,background:C.surface,color:C.text,fontSize:12,fontWeight:600,cursor:"pointer"}}>
        ← Rapor'a Dön
      </button>

      <div style={{marginBottom:20}}>
        <div style={{fontSize:16,fontWeight:700,color:C.text,marginBottom:4}}>Görsel: {result.stem}</div>
        <div style={{fontSize:12,color:C.textMuted}}>Oturum: {result.setId} · Durum: {result.status}</div>
      </div>

      {/* Bbox Toggle Buttons */}
      <div style={{display:"flex",gap:10,marginBottom:14,flexWrap:"wrap"}}>
        <button onClick={()=>setShowGt(v=>!v)}
          style={{padding:"6px 14px",borderRadius:6,border:`1px solid ${C.gt}44`,background:showGt?`${C.gt}22`:"transparent",color:C.gt,fontSize:12,cursor:"pointer",fontWeight:500}}>
          {showGt?"✓ GT":"GT"} Göster
        </button>
        <button onClick={()=>setShowPhysician(v=>!v)}
          style={{padding:"6px 14px",borderRadius:6,border:`1px solid ${C.physician}44`,background:showPhysician?`${C.physician}22`:"transparent",color:C.physician,fontSize:12,cursor:"pointer",fontWeight:500}}>
          {showPhysician?"✓ Hekim":"Hekim"} Göster
        </button>
        {imageData?.pred?.length>0&&(
          <button onClick={()=>setShowPred(v=>!v)}
            style={{padding:"6px 14px",borderRadius:6,border:`1px solid ${C.ai}44`,background:showPred?`${C.ai}22`:"transparent",color:C.ai,fontSize:12,cursor:"pointer",fontWeight:500}}>
            {showPred?"✓ AI":"AI"} Göster
          </button>
        )}
      </div>

      {imageData&&(
        <div style={{marginBottom:20,borderRadius:12,border:`1px solid ${C.border}`,overflow:"hidden",background:C.surface}}>
          <ComparisonCanvas 
            imageData={imageData} 
            physicianBoxes={result.boxes||[]} 
            gtYoloBoxes={imageData.gt||[]}
            predBoxes={imageData.pred||[]}
            showGt={showGt}
            showPhysician={showPhysician}
            showPred={showPred}
          />
        </div>
      )}

      {/* İstatistikler */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:10,marginBottom:20}}>
        <div style={{background:C.surfaceAlt,border:`1px solid ${C.border}`,borderRadius:8,padding:12}}>
          <div style={{fontSize:11,color:C.textMuted,marginBottom:6}}>GT Bbox Sayısı</div>
          <div style={{fontSize:20,fontWeight:700,color:C.gt}}>{result.gtBoxCount}</div>
        </div>
        <div style={{background:C.surfaceAlt,border:`1px solid ${C.border}`,borderRadius:8,padding:12}}>
          <div style={{fontSize:11,color:C.textMuted,marginBottom:6}}>Hekim Bbox Sayısı</div>
          <div style={{fontSize:20,fontWeight:700,color:C.physician}}>{result.physicianBoxCount}</div>
        </div>
        <div style={{background:C.surfaceAlt,border:`1px solid ${C.border}`,borderRadius:8,padding:12}}>
          <div style={{fontSize:11,color:C.textMuted,marginBottom:6}}>IoU Skoru</div>
          <div style={{fontSize:20,fontWeight:700,color:result.iou!==null?(result.iou>0.3?C.success:C.danger):C.textDim}}>
            {result.iou!==null?result.iou.toFixed(2):"—"}
          </div>
        </div>
      </div>

      {/* Doğruluk göstergesi */}
      <div style={{background:C.surfaceAlt,border:`1px solid ${C.border}`,borderRadius:8,padding:14,marginBottom:20}}>
        <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:8}}>
          <div style={{fontSize:12,fontWeight:600,color:C.text}}>Karar Doğruluğu</div>
          <Badge color={result.labelCorrect?C.success:C.danger}>{result.labelCorrect?"DOĞRU":"YANLIŞ"}</Badge>
        </div>
        <div style={{fontSize:12,color:C.textMuted,lineHeight:1.6}}>
          {result.gtBoxCount>0
            ?(result.physicianBoxCount>0&&result.iou!==null
              ?`GT patoloji tespit edildi · Hekim buldu · IoU: ${result.iou.toFixed(3)}`
              :result.physicianBoxCount===0
              ?`GT patoloji var ama hekim kaçırdı`
              :`Hekim hatalı işaretledi`)
            :result.physicianBoxCount===0
            ?`GT patoloji yok · Hekim doğru (true negative)`
            :`Hekim yanlış işaretledi (false positive)`}
        </div>
      </div>

      {/* Feedback alanı */}
      <div style={{background:C.surfaceAlt,border:`1px solid ${C.border}`,borderRadius:8,padding:14,marginBottom:14}}>
        <label style={{display:"block",fontSize:12,fontWeight:600,color:C.text,marginBottom:10}}>
          Geri Bildirim / Notlar
        </label>
        <textarea
          value={feedback}
          onChange={(e)=>setFeedback(e.target.value)}
          placeholder="Bu görselle ilgili yorumlarınız..."
          style={{
            width:"100%",
            minHeight:80,
            padding:10,
            borderRadius:6,
            border:`1px solid ${C.border}`,
            background:C.bg,
            color:C.text,
            fontFamily:"'DM Sans', sans-serif",
            fontSize:12,
            resize:"vertical",
            outline:"none"
          }}
          onFocus={(e)=>e.target.style.borderColor=C.accent}
          onBlur={(e)=>e.target.style.borderColor=C.border}
        />
        <div style={{fontSize:11,color:C.textMuted,marginTop:8}}>
          {feedback.length} karakter {submitted&&"· Kaydedildi"}
        </div>
      </div>

      <button onClick={handleSubmit}
        style={{width:"100%",padding:12,background:C.accent,color:"#fff",border:"none",borderRadius:6,fontSize:14,fontWeight:600,cursor:"pointer"}}>
        {submitted?"✓ Geri Bildirimi Güncelle":"Geri Bildirim Kaydet"}
      </button>
    </div>
  );
}

// ── Rapor ─────────────────────────────────────────────────────────────────
function ReportView({results,mode,onReset,sessionImgs,physicianId,onStartOther}){
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
  const [sessionFeedback,setSessionFeedback]=useState("");

  useEffect(()=>{
    setSaveStatus("saving");
    const stats=buildSessionStats(results);
    const payload={
      physicianId:physicianId||"anonymous",
      mode:mode,
      completedAt:new Date().toISOString(),
      stats:stats,
      sessionFeedback:sessionFeedback.trim()||null
    };
    fetch("/api/stats",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)})
      .then(r=>r.json())
      .then(d=>setSaveStatus(d.ok?"saved":"error"))
      .catch(()=>{
        const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});
        setSavedUrl(URL.createObjectURL(blob));
        setSaveStatus("download");
      });
  },[]);

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

      {/* Session Feedback */}
      <div style={{background:C.surfaceAlt,border:`1px solid ${C.border}`,borderRadius:8,padding:14,marginBottom:14}}>
        <label style={{display:"block",fontSize:12,fontWeight:600,color:C.text,marginBottom:10}}>
          Oturum Geri Bildirimi / Notlar
        </label>
        <textarea
          value={sessionFeedback}
          onChange={(e)=>setSessionFeedback(e.target.value)}
          placeholder="Bu oturum hakkında genel notlarınız..."
          style={{
            width:"100%",
            minHeight:80,
            padding:10,
            borderRadius:6,
            border:`1px solid ${C.border}`,
            background:C.bg,
            color:C.text,
            fontFamily:"'DM Sans', sans-serif",
            fontSize:12,
            resize:"vertical",
            outline:"none"
          }}
          onFocus={(e)=>e.target.style.borderColor=C.accent}
          onBlur={(e)=>e.target.style.borderColor=C.border}
        />
        <div style={{fontSize:11,color:C.textMuted,marginTop:8}}>
          {sessionFeedback.length} karakter
        </div>
      </div>

      {/* Buttons */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
        <button onClick={onReset} style={{padding:12,background:C.accentSoft,border:`1px solid ${C.accent}44`,borderRadius:8,color:C.accent,fontSize:13,fontWeight:600,cursor:"pointer"}}>
          Anasayfaya Dön
        </button>
        <button onClick={onStartOther} style={{padding:12,background:mode==="solo"?C.ai:C.physician,border:"none",borderRadius:8,color:"#fff",fontSize:13,fontWeight:600,cursor:"pointer"}}>
          {mode==="solo"?"🤖 AI Oturumu Başla":"👁 Solo Oturumu Başla"}
        </button>
      </div>
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
  const [physicianId, setPhysicianId] = useState(null);
  const [showNameModal, setShowNameModal] = useState(false);
  const [pendingMode, setPendingMode] = useState(null);
  const [showInstructions, setShowInstructions] = useState(false);
  const [showFullscreen, setShowFullscreen] = useState(false);

  // Image cache for pre-loading
  const imageCache = useImageCache(sessionImgs, imgIdx);

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
    // Show name modal instead of starting directly
    setPendingMode(m);
    setShowNameModal(true);
  };

  const handleNameSubmit=(name)=>{
    if (!manifest || !pendingMode) return;
    setPhysicianId(name);
    setShowNameModal(false);
    startSessionWithMode(pendingMode);
    setPendingMode(null);
  };

  // Convert pred YOLO boxes to editable canvas boxes with source="ai"
  const predToBoxes=(pred)=>(pred??[]).map((pb,i)=>({
    id: Date.now()+i,
    x: pb.cx - pb.w/2,
    y: pb.cy - pb.h/2,
    w: pb.w,
    h: pb.h,
    source: "ai",
    conf: pb.conf,
  }));

  const startSessionWithMode=(m)=>{
    if (!manifest) return;
    const targetSet = m === "solo" ? "A" : "B";
    const sessionImages = manifest.images.filter(img => img.setId === targetSet);
    setSessionImgs(sessionImages);
    setMode(m); 
    setImgIdx(0); 
    setResults([]); 
    // For AI mode: pre-populate first image's boxes with pred boxes
    setBoxes(m==="ai" ? predToBoxes(sessionImages[0]?.pred) : []); 
    setSelIdx(null);
    setStartTime(Date.now()); 
    setShowGt(false); 
    setShowPred(m==="ai");
    setPhase("session");
    setShowInstructions(true);
  };

  const startOtherSession=()=>{
    if (!manifest || !physicianId) return;
    const nextMode = mode === "solo" ? "ai" : "solo";
    startSessionWithMode(nextMode);
  };

  const submit=()=>{
    const elapsed=Date.now()-startTime;
    const gtBoxes=imageData?.gt??[];
    const iouScore=bestIou(boxes,gtBoxes);
    const hasGt=gtBoxes.length>0;

    let labelCorrect;
    if (hasGt) {
      labelCorrect = boxes.length > 0 && iouScore !== null && iouScore > 0.3;
    } else {
      labelCorrect = boxes.length === 0;
    }

    // AI interaction breakdown
    const aiAccepted   = boxes.filter(b=>b.source==="ai").length;
    const aiEdited     = boxes.filter(b=>b.source==="edited").length;
    const physicianNew = boxes.filter(b=>b.source==="physician").length;
    const predTotal    = imageData?.pred?.length??0;
    const aiDeleted    = Math.max(0, predTotal - aiAccepted - aiEdited);

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
      boxes:boxes.map(b=>({
        x:+b.x.toFixed(4),y:+b.y.toFixed(4),
        w:+b.w.toFixed(4),h:+b.h.toFixed(4),
        source:b.source??"physician",
        ...(b.conf!=null?{conf:b.conf}:{}),
      })),
      ...(mode==="ai"?{aiInteraction:{
        predCount:predTotal,
        accepted:aiAccepted,
        edited:aiEdited,
        deleted:aiDeleted,
        physicianAdded:physicianNew,
      }}:{}),
    };
    const newResults=[...results,rec];
    setResults(newResults);
    if(imgIdx+1>=sessionImgs.length){setPhase("report");}
    else{
      const nextIdx=imgIdx+1;
      setImgIdx(nextIdx);
      setSelIdx(null);
      setBoxes(mode==="ai" ? predToBoxes(sessionImgs[nextIdx]?.pred) : []);
      setStartTime(Date.now()); setShowGt(false); setShowPred(mode==="ai");
    }
  };

  const reset=()=>{setPhase("home");setMode(null);setPhysicianId(null);setImgIdx(0);setResults([]);setBoxes([]);setSelIdx(null);setSessionImgs([]);setPendingMode(null);setShowInstructions(false);setShowFullscreen(false);};

  return (
    <div style={{minHeight:"100vh",background:C.bg,color:C.text,fontFamily:"'DM Sans','Segoe UI',sans-serif",paddingBottom:48}}>
      {showNameModal && <NameInputModal onStart={handleNameSubmit} />}
      {showInstructions && <SessionInstructionModal mode={mode} onAcknowledge={()=>setShowInstructions(false)} />}
      {showFullscreen && <FullscreenViewer imageData={imageData} boxes={boxes} selectedIdx={selIdx} onBoxesChange={setBoxes} onSelectIdx={setSelIdx} showGt={showGt} showPred={showPred} onShowPredToggle={()=>setShowPred(v=>!v)} aiMode={mode==="ai"} mode={mode} imgIdx={imgIdx} sessionLength={sessionImgs.length} onSubmit={submit} onClose={()=>setShowFullscreen(false)} />}

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
              <button onClick={()=>setShowFullscreen(true)}
                style={{padding:"6px 14px",borderRadius:6,border:`1px solid ${C.border}`,background:"transparent",color:C.text,fontSize:12,cursor:"pointer",fontWeight:500}}>
                ⛶ Tam Ekran
              </button>
              <button onClick={submit}
                style={{padding:"6px 16px",borderRadius:6,border:"none",background:C.accent,color:"#fff",fontSize:12,fontWeight:600,cursor:"pointer"}}>
                {imgIdx+1>=sessionImgs.length?"Oturumu Bitir":"Sonraki →"}
              </button>
            </div>

            <AnnotationCanvas
              imageData={imageData} boxes={boxes} selectedIdx={selIdx}
              onBoxesChange={setBoxes} onSelectIdx={setSelIdx}
              showGt={showGt} showPred={false}
              aiMode={mode==="ai"}
            />

            {/* Araç çubuğu */}
            <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginTop:8,flexWrap:"wrap",gap:6}}>
              <div style={{display:"flex",gap:5,alignItems:"center",flexWrap:"wrap"}}>
                {boxes.map((b,i)=>{
                  const isAi=b.source==="ai"||b.source==="edited";
                  const col=isAi?C.ai:C.physician;
                  return (
                    <button key={i} onClick={()=>setSelIdx(selIdx===i?null:i)}
                      style={{padding:"3px 9px",borderRadius:5,
                        border:`1px solid ${selIdx===i?col:C.border}`,
                        background:selIdx===i?(isAi?C.aiSoft:C.physicianSoft):"transparent",
                        color:selIdx===i?col:C.textMuted,
                        fontSize:11,cursor:"pointer",fontFamily:"monospace"}}>
                      {isAi?"AI":"●"} #{i+1}
                    </button>
                  );
                })}
                {!boxes.length&&<span style={{fontSize:11,color:C.textDim}}>
                  {mode==="ai"?"AI tahmini yok":"Bbox yok"}
                </span>}
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
                {mode==="ai"
                  ? boxes.length===0
                    ? <span>AI tahmini yok — boş bırakabilir veya yeni bbox çizebilirsiniz</span>
                    : <span style={{color:C.success}}>
                        ✓ {boxes.filter(b=>b.source==="ai").length} AI onaylandı
                        {boxes.filter(b=>b.source==="edited").length>0&&` · ${boxes.filter(b=>b.source==="edited").length} düzenlendi`}
                        {boxes.filter(b=>b.source==="physician").length>0&&` · ${boxes.filter(b=>b.source==="physician").length} yeni eklendi`}
                      </span>
                  : boxes.length===0
                    ? <span>Patoloji yoksa boş bırakabilirsiniz</span>
                    : <span style={{color:C.success}}>✓ {boxes.length} bbox — ilerleyebilirsiniz</span>
                }
              </div>
            </div>
          </div>
        )}

        {phase==="report"&&<ReportView results={results} mode={mode} onReset={reset} sessionImgs={sessionImgs} physicianId={physicianId} onStartOther={startOtherSession}/>}
      </div>
    </div>
  );
}
