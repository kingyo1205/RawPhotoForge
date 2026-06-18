extends Control

signal drag_started
signal drag_ended

var editor: Node = null
var dragging_index := -1

func set_editor(e) -> void:
	editor = e

func to_curve(p: Vector2) -> Vector2:
	var canvas := editor.get_node("Background/CurveCanvas") as Control
	var y_range = editor.get_y_range()
	var y_norm = clamp(1.0 - p.y / canvas.size.y, 0.0, 1.0)
	return Vector2(
		clamp(p.x / canvas.size.x, 0.0, 1.0),
		y_norm * (y_range.y - y_range.x) + y_range.x
	)

func _gui_input(event) -> void:
	if editor == null:
		return

	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			if event.pressed:
				_on_left_press(event.position)
			else:
				if dragging_index != -1:
					emit_signal("drag_ended")
				dragging_index = -1

		elif event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
			_on_right_press(event.position)

	elif event is InputEventMouseMotion and dragging_index != -1:
		_drag(event.position)

func _on_left_press(pos: Vector2) -> void:
	var idx := _find_point(pos)
	if idx != -1:
		dragging_index = idx
	else:
		var p := to_curve(pos)
		var insert_at = -1
		for i in range(editor.points.size()):
			if p.x < editor.points[i].x:
				insert_at = i
				break
		if insert_at == -1:
			insert_at = editor.points.size()
		
		editor.points.insert(insert_at, p)
		dragging_index = insert_at

	if dragging_index != -1:
		emit_signal("drag_started")

	editor.emit_curve()
	queue_redraw_all()

func _on_right_press(pos: Vector2) -> void:
	var idx := _find_point(pos)
	if idx > 0 and idx < editor.points.size() - 1:
		editor.points.remove_at(idx)
		editor.emit_curve()
	queue_redraw_all()

func _drag(pos: Vector2) -> void:
	var p := to_curve(pos)

	var min_x: float
	var max_x: float

	if dragging_index == 0:
		min_x = 0.0
	elif dragging_index > 0:
		min_x = editor.points[dragging_index - 1].x + 0.001
	
	if dragging_index == editor.points.size() - 1:
		max_x = 1.0
	elif dragging_index < editor.points.size() - 1:
		max_x = editor.points[dragging_index + 1].x - 0.001
	
	p.x = clamp(p.x, min_x, max_x)

	if dragging_index == 0:
		p.x = 0.0
	elif dragging_index == editor.points.size() - 1:
		p.x = 1.0
	
	var y_range = editor.get_y_range()
	p.y = clamp(p.y, y_range.x, y_range.y)

	editor.points[dragging_index] = p
	editor.emit_curve()
	queue_redraw_all()

func _find_point(pos: Vector2) -> int:
	var canvas := editor.get_node("Background/CurveCanvas") as Control
	for i in range(editor.points.size()):
		var sp: Vector2 = editor.to_screen(editor.points[i], canvas.size)
		if sp.distance_to(pos) <= 12.0:
			return i
	return -1

func queue_redraw_all() -> void:
	editor.get_node("Background/CurveCanvas").queue_redraw()
