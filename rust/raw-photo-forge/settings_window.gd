# settings_window.gd
extends Window

const SETTINGS_FILE_PATH = "user://settings.json"

@onready var ui_preview_size_slider: HSlider = %UIPreviewSizeSlider
@onready var ui_preview_size_line_edit: LineEdit = %UIPreviewSizeLineEdit
@onready var drag_preview_size_slider: HSlider = %DragPreviewSizeSlider
@onready var drag_preview_size_line_edit: LineEdit = %DragPreviewSizeLineEdit
@onready var info_dialog: AcceptDialog = %InfoDialog
@onready var language_option_button: OptionButton = %LanguageOptionButton

var settings: Dictionary = {}

func _ready() -> void:
	load_settings()
	# ウィンドウが閉じられたときに非表示にする
	close_requested.connect(hide)
	
	# UI要素の初期値を設定から反映
	_update_ui_from_settings()

	# スライダーとLineEditのシグナル接続
	ui_preview_size_slider.value_changed.connect(_on_ui_preview_size_slider_changed)
	ui_preview_size_line_edit.text_submitted.connect(_on_ui_preview_size_line_edit_submitted)
	drag_preview_size_slider.value_changed.connect(_on_drag_preview_size_slider_changed)
	drag_preview_size_line_edit.text_submitted.connect(_on_drag_preview_size_line_edit_submitted)

	var tab_container = $VBoxContainer/TabContainer
	tab_container.set_tab_title(0, tr("TR_SETTINGS_TAB_LANGUAGE"))
	tab_container.set_tab_title(1, tr("TR_SETTINGS_TAB_IMAGE"))

	# LanguageOptionButton setup
	language_option_button.add_item("English")
	language_option_button.set_item_metadata(0, "en")
	language_option_button.add_item("日本語")
	language_option_button.set_item_metadata(1, "ja")
	language_option_button.item_selected.connect(_on_language_selected)


func show_and_center(min_size: Vector2i = Vector2i(0, 0)) -> void:
	# 表示する前に最新の設定を読み込む
	load_settings()
	_update_ui_from_settings()
	super.popup_centered(min_size)

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
	
	var locale = settings.get("locale", TranslationServer.get_locale())
	TranslationServer.set_locale(locale)


func _load_default_settings() -> void:
	settings = {
		"image": {
			"ui_preview_size": 1280,
			"drag_preview_size": 400
		},
		"locale": "en"
	}

func _update_ui_from_settings() -> void:
	var image_settings = settings.get("image", {})
	var ui_preview_size = image_settings.get("ui_preview_size", 1280)
	var drag_preview_size = image_settings.get("drag_preview_size", 400)

	ui_preview_size_slider.value = ui_preview_size
	ui_preview_size_line_edit.text = str(ui_preview_size)
	drag_preview_size_slider.value = drag_preview_size
	drag_preview_size_line_edit.text = str(drag_preview_size)
	
	var locale = settings.get("locale", TranslationServer.get_locale())
	for i in range(language_option_button.item_count):
		if language_option_button.get_item_metadata(i) == locale:
			language_option_button.select(i)
			break


func save_settings() -> void:
	settings["image"] = {
		"ui_preview_size": int(ui_preview_size_line_edit.text),
		"drag_preview_size": int(drag_preview_size_line_edit.text)
	}
	settings["locale"] = TranslationServer.get_locale()
	
	var file = FileAccess.open(SETTINGS_FILE_PATH, FileAccess.WRITE)
	var json_string = JSON.stringify(settings, "	")
	file.store_string(json_string)

func _on_save_button_pressed() -> void:
	save_settings()
	info_dialog.dialog_text = tr("TR_SETTINGS_SAVED_INFO")
	info_dialog.popup_centered()

func _on_language_selected(index: int) -> void:
	var selected_locale = language_option_button.get_item_metadata(index)
	TranslationServer.set_locale(selected_locale)

func _on_ui_preview_size_slider_changed(value: float) -> void:
	ui_preview_size_line_edit.text = str(int(value))

func _on_ui_preview_size_line_edit_submitted(new_text: String) -> void:
	if new_text.is_valid_int():
		var value = clamp(int(new_text), 500, 2000)
		ui_preview_size_line_edit.text = str(value)
		ui_preview_size_slider.value = value
	else:
		# 不正な値の場合はスライダーの値に戻す
		ui_preview_size_line_edit.text = str(int(ui_preview_size_slider.value))

func _on_drag_preview_size_slider_changed(value: float) -> void:
	drag_preview_size_line_edit.text = str(int(value))

func _on_drag_preview_size_line_edit_submitted(new_text: String) -> void:
	if new_text.is_valid_int():
		var value = clamp(int(new_text), 100, 800)
		drag_preview_size_line_edit.text = str(value)
		drag_preview_size_slider.value = value
	else:
		# 不正な値の場合はスライダーの値に戻す
		drag_preview_size_line_edit.text = str(int(drag_preview_size_slider.value))
