extends Control

const SETTINGS_FILE_PATH = "user://settings.json"
var settings: Dictionary = {}

@onready var editor_full = $PhotoEditorFull
@onready var editor_mid  = $PhotoEditorMid
@onready var editor_low  = $PhotoEditorLow
@onready var gpu_processor = $GpuProcessor

@onready var tex_rect: TextureRect = \
	$VBoxContainer/HBoxContainer/ImageArea/TextureRect

@onready var exposure_label: Label = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/ExposureLabel
@onready var exposure_slider: HSlider = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/ExposureSlider

@onready var contrast_label: Label = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/ContrastLabel
@onready var contrast_slider: HSlider = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/ContrastSlider

@onready var shadow_label: Label = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/ShadowLabel
@onready var shadow_slider: HSlider = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/ShadowSlider

@onready var highlight_label: Label = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/HighlightLabel
@onready var highlight_slider: HSlider = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/HighlightSlider

@onready var black_label: Label = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/BlackLabel
@onready var black_slider: HSlider = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/BlackSlider

@onready var white_label: Label = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/WhiteLabel
@onready var white_slider: HSlider = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/WhiteSlider

@onready var brightness_tone_curve_editor: Control = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Brightness/VBoxContainer/ToneCurveEditor
@onready var hue_tone_curve_editor: Control = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Hue/VBoxContainer/ToneCurveEditor
@onready var saturation_tone_curve_editor: Control = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Saturation/VBoxContainer/ToneCurveEditor
@onready var lightness_tone_curve_editor: Control = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Lightness/VBoxContainer/ToneCurveEditor

@onready var temperature_label: Label = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/WB/ScrollContainer/VBoxContainer/TemperatureLabel
@onready var temperature_slider: HSlider = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/WB/ScrollContainer/VBoxContainer/TemperatureSlider

@onready var tint_label: Label = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/WB/ScrollContainer/VBoxContainer/TintLabel
@onready var tint_slider: HSlider = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/WB/ScrollContainer/VBoxContainer/TintSlider

@onready var vignette_label: Label = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Effect/ScrollContainer/VBoxContainer/VignetteLabel
@onready var vignette_slider: HSlider = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Effect/ScrollContainer/VBoxContainer/VignetteSlider

@onready var lens_distortion_label: Label = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Effect/ScrollContainer/VBoxContainer/LensDistortionLabel
@onready var lens_distortion_slider: HSlider = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Effect/ScrollContainer/VBoxContainer/LensDistortionSlider

@onready var metadata_tree: Tree = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Metadata/Tree

@onready var menu_bar: MenuBar = \
	$VBoxContainer/MenuBar

@onready var reset_tone_button: Button = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Tone/ScrollContainer/VBoxContainer/ResetToneButton
@onready var reset_wb_button: Button = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/WB/ScrollContainer/VBoxContainer/ResetWBButton
@onready var reset_effect_button: Button = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Effect/ScrollContainer/VBoxContainer/ResetEffectButton

@onready var reset_brightness_button: Button = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Brightness/VBoxContainer/ResetBrightnessButton
@onready var reset_hue_button: Button = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Hue/VBoxContainer/ResetHueButton
@onready var reset_saturation_button: Button = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Saturation/VBoxContainer/ResetSaturationButton
@onready var reset_lightness_button: Button = \
	$VBoxContainer/HBoxContainer/EditPanel/TabContainer/Lightness/VBoxContainer/ResetLightnessButton
	




@onready var file_dialog: FileDialog = $FileDialog

@onready var save_dialog: Window = $SaveDialog
@onready var format_option_button: OptionButton = $SaveDialog/VBoxContainer/HBoxContainer/FormatOptionButton
@onready var save_button: Button = $SaveDialog/VBoxContainer/HBoxContainer2/SaveButton
@onready var cancel_button: Button = $SaveDialog/VBoxContainer/HBoxContainer2/CancelButton
@onready var save_file_dialog: FileDialog = $SaveFileDialog
@onready var confirmation_dialog: ConfirmationDialog = $ConfirmationDialog
@onready var control: Control = $'.'
@onready var settings_window: Window = %SettingsWindow


var edit := {
	"exposure": 0.0,
	"contrast": 0.0,
	"shadow": 0.0,
	"highlight": 0.0,
	"black": 0.0,
	"white": 0.0,
	"temperature": 0.0,
	"tint": 0.0,
	"vignette": 0.0,
	"lens_distortion": 0.0,
	"brightness_tone_curve_points": [],
	"hue_tone_curve_points": [],
	"saturation_tone_curve_points": [],
	"lightness_tone_curve_points": [],
}

enum PreviewLevel { LOW, MID, FULL }
var preview_level := PreviewLevel.MID
var base_image: Image
var _image_loaded: bool = false
var _original_filename_base: String = ""


func _ready() -> void:
	load_settings()
	
	var adapters = gpu_processor.get_adapters()
	
	if not adapters:
		push_error("wgpu adapter zero")
		show_dialog(tr("TR_ERROR_NO_WGPU"))
		return

	var adapter_index = int(settings.get("wgpu_adapter", 0))
	
	print("adapter_index: %s" % str(adapter_index))
	print(adapters)
	if not gpu_processor.initialize(adapter_index):
		show_dialog(tr("TR_ERROR_GPU_INIT"))
		
		return
	
	
	for s in [
		exposure_slider, contrast_slider, shadow_slider,
		highlight_slider, black_slider, white_slider,
		temperature_slider, tint_slider, vignette_slider,
		lens_distortion_slider
	]:
		s.drag_started.connect(_on_drag_start)
		s.drag_ended.connect(_on_drag_end)

	exposure_slider.value_changed.connect(func(v): _on_value("exposure", v, exposure_label, "TR_EXPOSURE", false))
	contrast_slider.value_changed.connect(func(v): _on_value("contrast", v, contrast_label, "TR_CONTRAST", true))
	shadow_slider.value_changed.connect(func(v): _on_value("shadow", v, shadow_label, "TR_SHADOW", true))
	highlight_slider.value_changed.connect(func(v): _on_value("highlight", v, highlight_label, "TR_HIGHLIGHT", true))
	black_slider.value_changed.connect(func(v): _on_value("black", v, black_label, "TR_BLACK_LEVEL", true))
	white_slider.value_changed.connect(func(v): _on_value("white", v, white_label, "TR_WHITE_LEVEL", true))
	temperature_slider.value_changed.connect(func(v): _on_white_balance_changed("temperature", v, temperature_label, "TR_TEMPERATURE"))
	tint_slider.value_changed.connect(func(v): _on_white_balance_changed("tint", v, tint_label, "TR_TINT"))
	vignette_slider.value_changed.connect(func(v): _on_effect_value_changed("vignette", v, vignette_label, "TR_VIGNETTE"))
	lens_distortion_slider.value_changed.connect(func(v): _on_effect_value_changed("lens_distortion", v, lens_distortion_label, "TR_LENS_DISTORTION"))

	# MenuBar setup
	var file_menu = PopupMenu.new()
	file_menu.title = tr("TR_MENU_FILE")
	file_menu.name = "FileMenu"
	file_menu.add_item(tr("TR_MENU_OPEN"), 0)
	file_menu.add_item(tr("TR_MENU_SAVE"), 1)
	file_menu.id_pressed.connect(_on_file_menu_id_pressed)
	menu_bar.add_child(file_menu)

	var edit_menu = PopupMenu.new()
	edit_menu.title = tr("TR_MENU_EDIT")
	edit_menu.name = "EditMenu"
	edit_menu.add_item(tr("TR_MENU_RESET_ALL"), 0)
	edit_menu.id_pressed.connect(_on_edit_menu_id_pressed)
	menu_bar.add_child(edit_menu)

	var setting_menu = PopupMenu.new()
	setting_menu.title = tr("TR_MENU_SETTING_TITLE")
	setting_menu.name = "SettingMenu"
	setting_menu.add_item(tr("TR_MENU_SETTING"), 0)
	setting_menu.id_pressed.connect(_on_setting_menu_id_pressed)
	menu_bar.add_child(setting_menu)
	
	reset_tone_button.pressed.connect(_on_reset_tone_pressed)
	reset_wb_button.pressed.connect(_on_reset_wb_pressed)
	reset_effect_button.pressed.connect(_on_reset_effect_pressed)
	reset_brightness_button.pressed.connect(_on_reset_brightness_pressed)
	reset_hue_button.pressed.connect(_on_reset_hue_pressed)
	reset_saturation_button.pressed.connect(_on_reset_saturation_pressed)
	reset_lightness_button.pressed.connect(_on_reset_lightness_pressed)

	tex_rect.gui_input.connect(_on_image_input)
	
	file_dialog.file_selected.connect(_open_image)

	save_dialog.close_requested.connect(_on_save_dialog_cancel_pressed)
	save_button.pressed.connect(_on_save_dialog_save_pressed)
	cancel_button.pressed.connect(_on_save_dialog_cancel_pressed)
	save_file_dialog.file_selected.connect(_on_save_file_selected)

	format_option_button.add_item("JPEG")
	format_option_button.set_item_metadata(0, "jpeg")
	format_option_button.add_item("PNG")
	format_option_button.set_item_metadata(1, "png")

	_setup_curve_editor(brightness_tone_curve_editor, "brightness_tone_curve_points", 0, "res://assets/tone_curve/brightness_gradient.png")
	_setup_curve_editor(hue_tone_curve_editor, "hue_tone_curve_points", 1, "res://assets/tone_curve/hue_bars.png")
	_setup_curve_editor(saturation_tone_curve_editor, "saturation_tone_curve_points", 2, "res://assets/tone_curve/hue_vs_saturation.png")
	_setup_curve_editor(lightness_tone_curve_editor, "lightness_tone_curve_points", 3, "res://assets/tone_curve/hue_vs_lightness.png")

	call_deferred("_set_image_loaded", false)

	var tab_container = $VBoxContainer/HBoxContainer/EditPanel/TabContainer
	tab_container.set_tab_title(0, tr("TR_TAB_TONE"))
	tab_container.set_tab_title(1, tr("TR_TAB_BRIGHTNESS"))
	tab_container.set_tab_title(2, tr("TR_TAB_HUE"))
	tab_container.set_tab_title(3, tr("TR_TAB_SATURATION"))
	tab_container.set_tab_title(4, tr("TR_TAB_LIGHTNESS"))
	tab_container.set_tab_title(5, tr("TR_TAB_WB"))
	tab_container.set_tab_title(6, tr("TR_TAB_EFFECT"))
	tab_container.set_tab_title(7, tr("TR_TAB_METADATA"))
	
	_update_all_slider_labels()
	
	
func show_dialog(message):
	confirmation_dialog.dialog_text = message
	confirmation_dialog.popup_centered()
	confirmation_dialog.get_cancel_button().hide()
	


func load_settings() -> void:
	if FileAccess.file_exists(SETTINGS_FILE_PATH):
		var file = FileAccess.open(SETTINGS_FILE_PATH, FileAccess.READ)
		var content = file.get_as_text()
		var json = JSON.new()
		var error = json.parse(content)
		if error == OK:
			settings = json.get_data()
		else:
			print("Error parsing settings.json: ", json.get_error_message(), " at line ", json.get_error_line())
			_load_default_settings()
	else:
		_load_default_settings()
	
	var locale = settings.get("locale", "en")
	TranslationServer.set_locale(locale)


func _load_default_settings() -> void:
	settings = {
		"wgpu_adapter": 0,
		"image": {
			"ui_preview_size": 1280,
			"drag_preview_size": 400
		},
		"locale": "en"
	}


func _set_image_loaded(loaded: bool) -> void:
	_image_loaded = loaded


func _on_file_menu_id_pressed(id: int) -> void:
	if id == 0:
		_open_dialog()
	elif id == 1:
		_on_save_photo_pressed()


func _on_save_photo_pressed() -> void:
	if not _image_loaded:
		return
	format_option_button.select(0)
	save_dialog.popup_centered()


func _on_save_dialog_cancel_pressed() -> void:
	save_dialog.hide()


func _on_save_dialog_save_pressed() -> void:
	save_dialog.hide()
	var filters = PackedStringArray()
	var selected_format = format_option_button.get_item_metadata(format_option_button.selected)
	if selected_format == "jpeg":
		filters.append("*.jpeg;%s" % tr("TR_JPEG_IMAGE"))
	elif selected_format == "png":
		filters.append("*.png;%s" % tr("TR_PNG_IMAGE"))
	save_file_dialog.filters = filters
	
	
	save_file_dialog.current_file = "%s_edited.%s" % [_original_filename_base, selected_format]
	
	save_file_dialog.popup_centered()


func _on_save_file_selected(path: String) -> void:
	var format = format_option_button.get_item_text(format_option_button.selected)
	_set_editor_parameters(editor_full)
	editor_full.apply_adjustments()
	var data: PackedByteArray = editor_full.save(format)
	var file = FileAccess.open(path, FileAccess.WRITE)
	if file:
		file.store_buffer(data)
		file.close()
		show_dialog(tr("TR_SAVED_FILE") % path)
		


func _on_edit_menu_id_pressed(id: int) -> void:
	if id == 0:
		_reset_all_edits()


func _on_setting_menu_id_pressed(_id: int) -> void:
	if _id == 0:
		settings_window.show_and_center()


func _on_reset_tone_pressed() -> void:
	_reset_tone()


func _on_reset_wb_pressed() -> void:
	_reset_wb()


func _on_reset_effect_pressed() -> void:
	_reset_effect()


func _on_reset_brightness_pressed() -> void:
	_reset_tone_curve("brightness_tone_curve_points", brightness_tone_curve_editor)


func _on_reset_hue_pressed() -> void:
	_reset_tone_curve("hue_tone_curve_points", hue_tone_curve_editor)


func _on_reset_saturation_pressed() -> void:
	_reset_tone_curve("saturation_tone_curve_points", saturation_tone_curve_editor)


func _on_reset_lightness_pressed() -> void:
	_reset_tone_curve("lightness_tone_curve_points", lightness_tone_curve_editor)


func _reset_all_edits() -> void:
	_reset_tone()
	_reset_wb()
	_reset_effect()
	_reset_tone_curve("brightness_tone_curve_points", brightness_tone_curve_editor)
	_reset_tone_curve("hue_tone_curve_points", hue_tone_curve_editor)
	_reset_tone_curve("saturation_tone_curve_points", saturation_tone_curve_editor)
	_reset_tone_curve("lightness_tone_curve_points", lightness_tone_curve_editor)


func _reset_tone() -> void:
	edit["exposure"] = 0.0
	edit["contrast"] = 0.0
	edit["shadow"] = 0.0
	edit["highlight"] = 0.0
	edit["black"] = 0.0
	edit["white"] = 0.0
	exposure_slider.value = 0.0
	contrast_slider.value = 0.0
	shadow_slider.value = 0.0
	highlight_slider.value = 0.0
	black_slider.value = 0.0
	white_slider.value = 0.0
	_update_image()


func _reset_wb() -> void:
	edit["temperature"] = 0.0
	edit["tint"] = 0.0
	temperature_slider.value = 0.0
	tint_slider.value = 0.0
	_update_image()


func _reset_effect() -> void:
	edit["vignette"] = 0.0
	edit["lens_distortion"] = 0.0
	vignette_slider.value = 0.0
	lens_distortion_slider.value = 0.0
	_update_image()


func _reset_tone_curve(key: String, editor: Control) -> void:
	editor.initialize_points()
	edit[key] = editor.points
	editor.emit_curve()
	_update_image()


func _setup_curve_editor(editor: Control, points_key: String, mode: int, texture_path: String) -> void:
	editor.mode = mode
	editor.get_node("Background").texture = load(texture_path)
	editor.curve_changed.connect(func(_lut): _on_curve_changed(editor.points, points_key))
	editor.drag_started.connect(_on_drag_start)
	editor.drag_ended.connect(_on_drag_end)
	editor.emit_curve()


func _on_curve_changed(points: Array, key: String) -> void:
	edit[key] = points
	_update_image()
	
	
func _open_dialog():
	file_dialog.popup()
	
func _open_image(path: String):
	_load_image(path)


func _load_image(path: String):
	var bytes = FileAccess.get_file_as_bytes(
		path
	)

	_original_filename_base = path.get_file().get_basename()
	var _original_file_ext = path.get_file().get_extension()
	
	if not editor_full.open_image(gpu_processor, bytes, _original_file_ext):
		show_dialog(tr("TR_ERROR_IMAGE_LOAD") % path)
		
		return
		
	base_image = editor_full.get_image()

	_init_preview_editors()
	_update_image()
	_update_metadata()
	_set_image_loaded(true)
	_update_all_slider_labels()


func _update_metadata():
	metadata_tree.clear()
	var root = metadata_tree.create_item()
	var exif = editor_full.get_exif()
	for key in exif.keys():
		var item = metadata_tree.create_item(root)
		item.set_text(0, key)
		item.set_text(1, str(exif[key]))
	
	
func _init_preview_editors() -> void:
	var image_settings = settings.get("image", {})
	var ui_preview_size = image_settings.get("ui_preview_size", 1280)
	var drag_preview_size = image_settings.get("drag_preview_size", 400)
	_open_resized(editor_mid, ui_preview_size)
	_open_resized(editor_low, drag_preview_size)


func _open_resized(editor, long_edge: int) -> void:
	var img := base_image.duplicate()
	var p_scale: float = float(long_edge) / max(img.get_width(), img.get_height())
	img.resize(
		int(img.get_width() * p_scale),
		int(img.get_height() * p_scale),
		Image.INTERPOLATE_BILINEAR
	)
	editor.open_image(gpu_processor, img.save_png_to_buffer(), "png")
	


func _update_all_slider_labels() -> void:
	# Tone
	_on_value("exposure", edit["exposure"], exposure_label, "TR_EXPOSURE", false)
	_on_value("contrast", edit["contrast"], contrast_label, "TR_CONTRAST", true)
	_on_value("shadow", edit["shadow"], shadow_label, "TR_SHADOW", true)
	_on_value("highlight", edit["highlight"], highlight_label, "TR_HIGHLIGHT", true)
	_on_value("black", edit["black"], black_label, "TR_BLACK_LEVEL", true)
	_on_value("white", edit["white"], white_label, "TR_WHITE_LEVEL", true)
	# WB
	_on_white_balance_changed("temperature", edit["temperature"], temperature_label, "TR_TEMPERATURE")
	_on_white_balance_changed("tint", edit["tint"], tint_label, "TR_TINT")
	# Effect
	_on_effect_value_changed("vignette", edit["vignette"], vignette_label, "TR_VIGNETTE")
	_on_effect_value_changed("lens_distortion", edit["lens_distortion"], lens_distortion_label, "TR_LENS_DISTORTION")


func _on_value(key: String, v: float, label: Label, p_name: String, is_int: bool) -> void:
	edit[key] = v
	if is_int:
		label.text = "%s %s" % [tr(p_name), str(int(v))]
	else:
		label.text = "%s %.2f" % [tr(p_name), v]
	_update_image()


func _on_white_balance_changed(key: String, v: float, label: Label, p_name: String) -> void:
	edit[key] = v
	label.text = "%s %s" % [tr(p_name), str(int(v))]
	_update_image()


func _on_effect_value_changed(key: String, v: float, label: Label, p_name: String) -> void:
	edit[key] = v
	label.text = "%s %s" % [tr(p_name), str(int(v))]
	_update_image()


func _on_drag_start() -> void:
	preview_level = PreviewLevel.LOW
	_update_image()


func _on_drag_end(_value_changed := true) -> void:
	preview_level = PreviewLevel.MID
	_update_image()


func _on_image_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		if event.pressed:
			if base_image:
				tex_rect.texture = ImageTexture.create_from_image(base_image)
		else:
			preview_level = PreviewLevel.MID
			_update_image()


func _update_image() -> void:
	if not _image_loaded:
		return
		
	var e = _current_editor()
	
	_set_editor_parameters(e)
	
	e.apply_adjustments()
	tex_rect.texture = ImageTexture.create_from_image(e.get_image())


func _set_editor_parameters(e) -> void:
	e.set_tone(
		edit["exposure"],
		edit["contrast"],
		edit["shadow"],
		edit["highlight"],
		edit["black"],
		edit["white"],
		""
		)
	e.set_whitebalance(
		edit["temperature"],
		edit["tint"],
		""
	)
	e.set_vignette(edit["vignette"])
	e.set_lens_distortion_correction(edit["lens_distortion"])
		
	var brightness_points = PackedVector2Array()
	for p in edit["brightness_tone_curve_points"]:
		brightness_points.append(Vector2(round(p.x * 65535.0), round(p.y * 65535.0)))
	e.set_brightness_tone_curve_from_points(brightness_points, "")

	var hue_points = PackedVector2Array()
	for p in edit["hue_tone_curve_points"]:
		hue_points.append(Vector2(round(p.x * 65535.0), round(p.y * 65535.0)))
	e.set_oklch_hue_curve_from_points(hue_points, "")
	
	var saturation_points = PackedVector2Array()
	for p in edit["saturation_tone_curve_points"]:
		saturation_points.append(Vector2(round(p.x * 65535.0), round(p.y / 2 * 65535.0)))
	e.set_oklch_saturation_curve_from_points(saturation_points, "")
	
	var lightness_points = PackedVector2Array()
	for p in edit["lightness_tone_curve_points"]:
		lightness_points.append(Vector2(round(p.x * 65535.0), round(p.y / 2 * 65535.0)))
	e.set_oklch_lightness_curve_from_points(lightness_points, "")


func _current_editor():
	match preview_level:
		PreviewLevel.LOW:  return editor_low
		PreviewLevel.MID:  return editor_mid
		_:                 return editor_full
