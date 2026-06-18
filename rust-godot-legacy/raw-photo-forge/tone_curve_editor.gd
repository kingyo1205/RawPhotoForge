extends Control

enum CurveMode {
	BRIGHTNESS,
	HUE,
	SATURATION,
	LIGHTNESS,
}

@export var mode: CurveMode = CurveMode.BRIGHTNESS:
	set(value):
		mode = value
		if is_inside_tree():
			initialize_points()

var points: Array[Vector2] = []

signal curve_changed(lut: PackedFloat32Array)
signal drag_started
signal drag_ended

func _ready() -> void:
	initialize_points()
	$Background/InputCatcher.set_editor(self)
	$Background/CurveCanvas.set_editor(self)
	$Background/InputCatcher.drag_started.connect(func(): emit_signal("drag_started"))
	$Background/InputCatcher.drag_ended.connect(func(): emit_signal("drag_ended"))

func initialize_points() -> void:
	if mode == CurveMode.BRIGHTNESS or mode == CurveMode.HUE:
		points = [Vector2(0.0, 0.0), Vector2(1.0, 1.0)]
	elif mode == CurveMode.SATURATION or mode == CurveMode.LIGHTNESS:
		points = [Vector2(0.0, 1.0), Vector2(1.0, 1.0)]
	
	if has_node("Background/CurveCanvas"):
		get_node("Background/CurveCanvas").queue_redraw()

func emit_curve() -> void:
	var lut := sample_curve(256)
	emit_signal("curve_changed", lut)

func sample_curve(n: int) -> PackedFloat32Array:
	var xs := PackedFloat32Array()
	var ys := PackedFloat32Array()
	for p in points:
		xs.append(p.x)
		ys.append(p.y)

	var result := PackedFloat32Array()
	for i in range(n):
		var x := float(i) / float(n - 1)
		result.append(pchip(xs, ys, x))
	return result

func get_y_range() -> Vector2:
	if mode == CurveMode.SATURATION or mode == CurveMode.LIGHTNESS:
		return Vector2(0.0, 2.0)
	else:
		return Vector2(0.0, 1.0)

func to_screen(p: Vector2, target_size: Vector2) -> Vector2:
	var y_range = get_y_range()
	return Vector2(
		p.x * target_size.x,
		(1.0 - (p.y - y_range.x) / (y_range.y - y_range.x)) * target_size.y
	)
	
	
func pchip(xs: PackedFloat32Array, ys: PackedFloat32Array, x: float) -> float:
	var n := xs.size()
	if n < 2:
		return 0.0

	# --- 1. 区間探索 ---
	
	if x <= xs[0]: return ys[0]
	if x >= xs[n - 1]: return ys[n - 1]

	var i := -1
	for j in range(n - 1):
		if x >= xs[j] and x < xs[j + 1]:
			i = j
			break
	
	
	if i == -1: i = n - 2

	# --- 2. スロープ（傾き）の計算 ---
	
	
	var m0: float
	var m1: float
	
	# m0 (左側の点の傾き)
	if i == 0:
		m0 = (ys[1] - ys[0]) / (xs[1] - xs[0])
	else:
		m0 = _get_pchip_slope_harmonic(
			xs[i] - xs[i-1], ys[i] - ys[i-1], 
			xs[i+1] - xs[i], ys[i+1] - ys[i]
		)

	# m1 (右側の点の傾き)
	if i == n - 2:
		m1 = (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2])
	else:
		m1 = _get_pchip_slope_harmonic(
			xs[i+1] - xs[i], ys[i+1] - ys[i], 
			xs[i+2] - xs[i+1], ys[i+2] - ys[i+1]
		)

	# --- 3. 補間計算 ---
	var x0 := xs[i]
	var x1 := xs[i + 1]
	var y0 := ys[i]
	var y1 := ys[i + 1]

	var h := x1 - x0
	# ゼロ除算防止
	if h <= 0.0: return y0
	
	var t := (x - x0) / h
	var t2 := t * t
	var t3 := t2 * t

	# エルミート基底関数
	var h00 := 2.0 * t3 - 3.0 * t2 + 1.0
	var h10 := t3 - 2.0 * t2 + t
	var h01 := -2.0 * t3 + 3.0 * t2
	var h11 := t3 - t2
	
	# 結果
	return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


func _get_pchip_slope_harmonic(dx1: float, dy1: float, dx2: float, dy2: float) -> float:
	var s1 := dy1 / dx1
	var s2 := dy2 / dx2
	
	# 符号が異なる（山や谷）の場合は傾き0（単調性維持のため）
	if s1 * s2 <= 0.0:
		return 0.0
	
	var w1 := 2.0 * dx2 + dx1
	var w2 := dx2 + 2.0 * dx1
	
	# Rust: (w1 + w2) / (w1 / s1 + w2 / s2)
	# 計算順序も合わせることで浮動小数点の誤差を最小化
	return (w1 + w2) / (w1 / s1 + w2 / s2)
