/* =====================================================
   Canvas setup
===================================================== */
const canvas = new fabric.Canvas('canvas', {
  selection: false, 
  backgroundColor: '#1a1a1a'
});

let img = null;
let drawMode = false;
let isDrawing = false;
let rect = null;
let startX, startY;
let currentFilename = "";

/* =====================================================
   Cau hinh Handles (Diem nắm)
===================================================== */
fabric.Object.prototype.set({
  cornerColor: '#00e5ff',
  cornerStrokeColor: '#003f46',
  cornerStyle: 'circle',
  cornerSize: 10,
  transparentCorners: false,
  borderColor: '#ff4757',
  borderScaleFactor: 2,
  objectCaching: false
});

/* =====================================================
   Upload image
===================================================== */
document.getElementById('upload').onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;

  currentFilename = file.name;
  canvas.clear();
  document.getElementById('output').textContent = "";

  const reader = new FileReader();
  reader.onload = (f) => {
    fabric.Image.fromURL(f.target.result, (image) => {
      img = image;

      canvas.setWidth(image.width);
      canvas.setHeight(image.height);

      image.set({
        selectable: false,
        evented: false
      });

      canvas.setBackgroundImage(image, canvas.renderAll.bind(canvas));
    });
  };
  reader.readAsDataURL(file);
};

/* =====================================================
   Toggle Draw Mode (Dung text khong dau de tranh loi font)
===================================================== */
const drawBtn = document.getElementById('drawBtn');
drawBtn.onclick = () => {
  drawMode = !drawMode;
  // Thay đổi text không dấu để an toàn tuyệt đối
  drawBtn.innerText = drawMode ? 'Draw Mode: ON' : 'Draw Mode: OFF';
  drawBtn.style.background = drawMode ? '#f59e0b' : '#4f46e5';

  canvas.discardActiveObject();
  canvas.renderAll();
};

/* =====================================================
   Mouse events: Ve Box xuyen thau
===================================================== */
canvas.on('mouse:down', (opt) => {
  if (!img || !drawMode) return;

  // Neu click trung vien (target), cho phep resize/move chu khong ve moi
  if (opt.target) return; 

  isDrawing = true;
  const p = canvas.getPointer(opt.e);
  startX = p.x;
  startY = p.y;

  rect = new fabric.Rect({
    left: startX,
    top: startY,
    width: 0,
    height: 0,
    fill: null,               // Trong suot de ve chong len nhau
    stroke: '#ff4757',
    strokeWidth: 4,
    selectable: true,
    hasControls: true,
    hasBorders: true,
    perPixelTargetFind: true, // Chi bat dinh khi click vao vien
    targetFindTolerance: 10   
  });

  canvas.add(rect);
  canvas.setActiveObject(rect);
});

canvas.on('mouse:move', (opt) => {
  if (!isDrawing || !rect) return;

  const p = canvas.getPointer(opt.e);

  rect.set({
    left: Math.min(p.x, startX),
    top: Math.min(p.y, startY),
    width: Math.abs(p.x - startX),
    height: Math.abs(p.y - startY)
  });

  canvas.renderAll();
});

canvas.on('mouse:up', () => {
  isDrawing = false;
  rect = null;
});

/* =====================================================
   Phim tat & Export
===================================================== */
document.addEventListener('keydown', (e) => {
  if (e.key === 'Delete' || e.key === 'Backspace') {
    const obj = canvas.getActiveObject();
    if (obj) canvas.remove(obj);
  }
});

document.getElementById('submitBtn').onclick = async () => {
  const boxes = canvas.getObjects('rect').map(r => ({
    xmin: Math.round(r.left),
    ymin: Math.round(r.top),
    xmax: Math.round(r.left + r.width * r.scaleX),
    ymax: Math.round(r.top + r.height * r.scaleY)
  }));

  // payload để gửi data về
  const payload = {
    image_name: currentFilename, 
    boxes: boxes
  };

  // 1. Hiển thị preview lên màn hình
  document.getElementById('output').textContent = JSON.stringify(payload, null, 2);

  // 2. Gửi về Backend
  try {
    const response = await fetch('http://127.0.0.1:8001/submit_boxes', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      const result = await response.json();
      alert(`Saved ${result.num_boxes} boxes to server!`);
    } else {
      alert('Error when saving boxes to server.');
    }
  } catch (error) {
    console.error('Connection error:', error);
    alert('Cannot connect to backend, ensure that FastApi is running.');
  }
};

/* Event highlight khi chon */
canvas.on('selection:created', handleSelect);
canvas.on('selection:updated', handleSelect);
canvas.on('selection:cleared', handleDeselect);

function handleSelect(e) {
  const obj = e.selected?.[0];
  if (!obj) return;
  obj.set({ strokeWidth: 5, cornerColor: '#00ff6a', cornerSize: 14 });
  canvas.requestRenderAll();
}

function handleDeselect(e) {
  e.deselected?.forEach(obj => {
    obj.set({ strokeWidth: 4, cornerColor: '#00e5ff', cornerSize: 10 });
  });
  canvas.requestRenderAll();
}