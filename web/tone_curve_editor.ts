import { pchipInterpolate } from "./core/interpolation.ts";

export enum CurveMode {
    BRIGHTNESS,
    HUE,
    SATURATION,
    LIGHTNESS,
}

export type Point = { x: number; y: number };

// タッチ操作しやすいように当たり判定を少し大きくする
const POINT_RADIUS = 8.0;

export class ToneCurveEditor {
    private container: HTMLElement;
    private mode: CurveMode;
    public points: Point[] = [];
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private draggingIndex = -1;

    // タッチ操作用
    private lastTap = 0;
    private touchIdentifier: number | null = null;
    private hasMoved = false;

    private dispatchCurveChange: () => void;
    private dispatchDragStart: () => void;
    private dispatchDragEnd: () => void;

    constructor(
        containerId: string,
        mode: CurveMode,
        onCurveChange: (points: Point[]) => void,
        onDragStart: () => void,
        onDragEnd: () => void
    ) {
        this.container = document.getElementById(containerId)!;
        this.mode = mode;
        this.dispatchCurveChange = () => onCurveChange(this.points);
        this.dispatchDragStart = onDragStart;
        this.dispatchDragEnd = onDragEnd;

        this.canvas = document.createElement('canvas');
        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d')!;

        this.initializePoints();
        this.setupCanvas();
        this.addEventListeners();
        this.draw();
    }

    public setBackground(imagePath: string) {
        this.container.style.backgroundImage = `url(${imagePath})`;
    }

    public initializePoints(): void {
        if (this.mode === CurveMode.BRIGHTNESS || this.mode === CurveMode.HUE) {
            this.points = [{ x: 0.0, y: 0.0 }, { x: 1.0, y: 1.0 }];
        } else if (this.mode === CurveMode.SATURATION || this.mode === CurveMode.LIGHTNESS) {
            this.points = [{ x: 0.0, y: 1.0 }, { x: 1.0, y: 1.0 }];
        }
        this.draw();
    }

    private setupCanvas(): void {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.container.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        this.canvas.style.width = `${rect.width}px`;
        this.canvas.style.height = `${rect.height}px`;
    }

    private addEventListeners(): void {
        // Mouse Events
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onPointerUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.onPointerUp.bind(this));
        this.canvas.addEventListener('dblclick', this.onDoubleClick.bind(this));

        // Touch Events
        this.canvas.addEventListener('touchstart', this.onTouchStart.bind(this), { passive: false });
        this.canvas.addEventListener('touchmove', this.onTouchMove.bind(this), { passive: false });
        this.canvas.addEventListener('touchend', this.onTouchEnd.bind(this));
        this.canvas.addEventListener('touchcancel', this.onPointerUp.bind(this));

        // Context Menu
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    private getYRange(): { min: number, max: number } {
        if (this.mode === CurveMode.SATURATION || this.mode === CurveMode.LIGHTNESS) {
            return { min: 0.0, max: 2.0 };
        }
        return { min: 0.0, max: 1.0 };
    }

    private toScreen(p: Point): Point {
        const rect = this.canvas.getBoundingClientRect();
        const yRange = this.getYRange();
        return {
            x: p.x * rect.width,
            y: (1.0 - (p.y - yRange.min) / (yRange.max - yRange.min)) * rect.height
        };
    }

    private toCurve(p: Point): Point {
        const rect = this.canvas.getBoundingClientRect();
        const yRange = this.getYRange();
        const yNorm = Math.max(0, Math.min(1, 1.0 - p.y / rect.height));
        return {
            x: Math.max(0, Math.min(1, p.x / rect.width)),
            y: yNorm * (yRange.max - yRange.min) + yRange.min
        };
    }

    private getPointerPosition(event: MouseEvent | Touch): Point {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
    }

    private findPoint(pos: Point): number {
        for (let i = 0; i < this.points.length; i++) {
            const sp = this.toScreen(this.points[i]);
            const distance = Math.sqrt(Math.pow(sp.x - pos.x, 2) + Math.pow(sp.y - pos.y, 2));
            if (distance <= POINT_RADIUS * 1.5) { // タッチしやすいように当たり判定を拡大
                return i;
            }
        }
        return -1;
    }

    private onPointerDown(pos: Point, isRightClick: boolean = false): void {
        if (isRightClick) {
            this.deletePointAt(pos);
            return;
        }

        const idx = this.findPoint(pos);
        if (idx !== -1) {
            this.draggingIndex = idx;
        } else {
            const p = this.toCurve(pos);
            let insertAt = this.points.findIndex(pt => pt.x > p.x);
            if (insertAt === -1) {
                insertAt = this.points.length;
            }
            this.points.splice(insertAt, 0, p);
            this.draggingIndex = insertAt;
        }
        this.dispatchDragStart();
        this.emitCurveChange();
    }

    private onPointerMove(pos: Point): void {
        if (this.draggingIndex === -1) return;
        this.hasMoved = true;

        const p = this.toCurve(pos);

        const minX = this.draggingIndex > 0 ? this.points[this.draggingIndex - 1].x + 0.001 : 0.0;
        const maxX = this.draggingIndex < this.points.length - 1 ? this.points[this.draggingIndex + 1].x - 0.001 : 1.0;

        p.x = Math.max(minX, Math.min(maxX, p.x));

        if (this.draggingIndex === 0) p.x = 0.0;
        if (this.draggingIndex === this.points.length - 1) p.x = 1.0;

        const yRange = this.getYRange();
        p.y = Math.max(yRange.min, Math.min(yRange.max, p.y));

        this.points[this.draggingIndex] = p;
        this.emitCurveChange();
    }

    private onPointerUp(): void {
        if (this.draggingIndex !== -1) {
            this.draggingIndex = -1;
            this.dispatchDragEnd();
        }
        this.touchIdentifier = null;
        this.hasMoved = false;
    }

    private deletePointAt(pos: Point): void {
        const idx = this.findPoint(pos);
        if (idx > 0 && idx < this.points.length - 1) {
            this.points.splice(idx, 1);
            this.emitCurveChange();
        }
    }

    private onMouseDown(event: MouseEvent): void {
        const pos = this.getPointerPosition(event);
        this.onPointerDown(pos, event.button === 2);
    }

    private onMouseMove(event: MouseEvent): void {
        if (event.buttons !== 1) return;
        const pos = this.getPointerPosition(event);
        this.onPointerMove(pos);
    }

    private onDoubleClick(event: MouseEvent): void {
        const pos = this.getPointerPosition(event);
        this.deletePointAt(pos);
    }

    private onTouchStart(event: TouchEvent): void {
        event.preventDefault();
        if (this.touchIdentifier !== null) return;

        const touch = event.changedTouches[0];
        this.touchIdentifier = touch.identifier;
        const pos = this.getPointerPosition(touch);
        
        const currentTime = new Date().getTime();
        const tapLength = currentTime - this.lastTap;

        if (tapLength < 300 && tapLength > 0) {
            this.deletePointAt(pos);
            this.lastTap = 0; // ダブルタップ後はリセット
        } else {
            this.onPointerDown(pos);
        }
        this.lastTap = currentTime;
    }

    private onTouchMove(event: TouchEvent): void {
        event.preventDefault();
        if (this.draggingIndex === -1) return;

        const touch = Array.from(event.changedTouches).find(t => t.identifier === this.touchIdentifier);
        if (!touch) return;

        const pos = this.getPointerPosition(touch);
        this.onPointerMove(pos);
    }

    private onTouchEnd(event: TouchEvent): void {
        event.preventDefault();
        const touch = Array.from(event.changedTouches).find(t => t.identifier === this.touchIdentifier);
        if (!touch) return;
        
        this.onPointerUp();
    }

    private emitCurveChange(): void {
        this.draw();
        this.dispatchCurveChange();
    }

    public sampleCurve(n: number): Float32Array {
        const xPts = this.points.map(p => p.x);
        const yPts = this.points.map(p => p.y);
        const xEval = new Float32Array(n).map((_, i) => i / (n - 1));
        return pchipInterpolate(xPts, yPts, xEval);
    }

    public draw(): void {
        requestAnimationFrame(() => {
            this.setupCanvas();
            const rect = this.container.getBoundingClientRect();
            this.ctx.clearRect(0, 0, rect.width, rect.height);
            this.drawCurve();
            this.drawPoints();
        });
    }

    private drawCurve(): void {
        const lut = this.sampleCurve(256);
        this.ctx.beginPath();

        let p = this.toScreen({ x: 0, y: lut[0] });
        this.ctx.moveTo(p.x, p.y);

        for (let i = 1; i < lut.length; i++) {
            p = this.toScreen({ x: i / (lut.length - 1), y: lut[i] });
            this.ctx.lineTo(p.x, p.y);
        }

        this.ctx.strokeStyle = 'blue';
        this.ctx.lineWidth = 2.0;
        this.ctx.stroke();
    }

    private drawPoints(): void {
        for (let i = 0; i < this.points.length; i++) {
            const p = this.points[i];
            const screenP = this.toScreen(p);

            this.ctx.beginPath();
            this.ctx.arc(screenP.x, screenP.y, POINT_RADIUS + 2.0, 0, 2 * Math.PI);
            this.ctx.fillStyle = 'black';
            this.ctx.fill();

            this.ctx.beginPath();
            this.ctx.arc(screenP.x, screenP.y, POINT_RADIUS, 0, 2 * Math.PI);
            this.ctx.fillStyle = 'red';
            this.ctx.fill();
        }
    }
}
