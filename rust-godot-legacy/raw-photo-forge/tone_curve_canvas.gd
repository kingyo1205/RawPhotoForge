extends Control

const POINT_RADIUS := 6.0
var editor: Node = null

func set_editor(e) -> void:
	editor = e

func _draw() -> void:
	if editor == null:
		return
	_draw_curve()
	_draw_points()

func _draw_curve() -> void:
	var lut: PackedFloat32Array = editor.sample_curve(256)
	var prev_p = Vector2(0.0, lut[0])
	var prev_screen_p = editor.to_screen(prev_p, size)

	for i in range(1, lut.size()):
		var p = Vector2(float(i) / (lut.size() - 1), lut[i])
		var screen_p = editor.to_screen(p, size)
		draw_line(prev_screen_p, screen_p, Color.BLUE, 2.0)
		prev_screen_p = screen_p

func _draw_points() -> void:
	for p in editor.points:
		var screen_p = editor.to_screen(p, size)
		draw_circle(screen_p, POINT_RADIUS + 1.0, Color.BLACK)
		draw_circle(screen_p, POINT_RADIUS, Color.RED)
