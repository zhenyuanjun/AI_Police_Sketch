"use client";

import { PointerEvent, useState } from "react";

export default function Face3DViewer() {
  const [rotation, setRotation] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [dragging, setDragging] = useState(false);
  const [lastX, setLastX] = useState(0);

  function startDrag(event: PointerEvent<HTMLDivElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
    setLastX(event.clientX);
  }

  function drag(event: PointerEvent<HTMLDivElement>) {
    if (!dragging) return;
    setRotation((value) => Math.max(-90, Math.min(90, value + (event.clientX - lastX) * 0.45)));
    setLastX(event.clientX);
  }

  function endDrag(event: PointerEvent<HTMLDivElement>) {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setDragging(false);
  }

  function zoomView(event: React.WheelEvent<HTMLDivElement>) {
    event.preventDefault();
    setZoom((value) => Math.max(0.82, Math.min(1.2, value - event.deltaY * 0.0008)));
  }

  function resetView() {
    setRotation(0);
    setZoom(1);
  }

  return <div className="relative overflow-hidden border border-[#d6d0c7] bg-[#ece8e1] shadow-inner" onPointerDown={startDrag} onPointerMove={drag} onPointerUp={endDrag} onPointerCancel={endDrag} onWheel={zoomView} style={{ touchAction: "none" }}>
    <div className="absolute left-5 top-5 z-10 text-[10px] uppercase tracking-[0.22em] text-[#8d867d]">3D view — prototype</div>
    <div className="flex min-h-[480px] items-center justify-center [perspective:900px]">
      <div className="relative h-[330px] w-[250px] transition-transform duration-75" style={{ transform: `scale(${zoom}) rotateY(${rotation}deg)`, transformStyle: "preserve-3d" }}>
        <div className="absolute inset-0 rounded-[48%_48%_44%_44%] border-[3px] border-[#6f6a63] bg-[#d8d2c9] shadow-[inset_18px_0_0_#e7e2da,inset_-18px_0_0_#c4bdb4]" style={{ transform: "translateZ(0)" }} />
        <div className="absolute left-[104px] top-[55px] h-[88px] w-[42px] rounded-[50%] border-t-2 border-[#8a8279]" style={{ transform: "translateZ(32px)" }} />
        <div className="absolute left-[55px] top-[112px] h-3 w-[54px] rounded-full bg-[#625e59]" style={{ transform: "translateZ(43px)" }} />
        <div className="absolute right-[55px] top-[112px] h-3 w-[54px] rounded-full bg-[#625e59]" style={{ transform: "translateZ(43px)" }} />
        <div className="absolute left-[76px] top-[139px] h-5 w-5 rounded-full bg-[#45433f] shadow-[74px_0_#45433f]" style={{ transform: "translateZ(47px)" }} />
        <div className="absolute left-[108px] top-[139px] h-[62px] w-[34px] border-l-2 border-r-2 border-[#837b72]" style={{ transform: "translateZ(48px)" }} />
        <div className="absolute bottom-[68px] left-[83px] h-4 w-[84px] rounded-[50%] border-b-2 border-[#65605b]" style={{ transform: "translateZ(46px)" }} />
        <div className="absolute -bottom-[30px] left-[-25px] h-[78px] w-[300px] rounded-[50%] border-t-[3px] border-[#77716a]" style={{ transform: "translateZ(3px)" }} />
      </div>
    </div>
    <div className="absolute bottom-4 left-5 right-5 flex items-end justify-between text-xs text-[#888077]"><span>Drag horizontally to rotate · Scroll to zoom</span><button type="button" onClick={resetView} className="border border-[#8d867d] px-3 py-1 uppercase tracking-[0.1em] hover:bg-white">Reset view</button></div>
    <p className="absolute right-5 top-10 text-[10px] text-[#8d867d]">Estimated / generic 3D visualization</p>
  </div>;
}
