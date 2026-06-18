use ndarray::Array1;
use photo_editor::{self, GpuProcessor};
use std::fs;
use std::io;
use std::{path::PathBuf, sync::Arc};

fn main() {
    let mut file_path = String::new();
    println!("Please enter the path");
    io::stdin().read_line(&mut file_path).unwrap();

    let mut file_pb = PathBuf::from(file_path.trim());

    println!("file path: {:?}", file_pb);

    println!("adapter info:");
    for adapter in GpuProcessor::get_adapter_list() {
        println!("{:?}", adapter.get_info());
        println!("{:?}", adapter.limits())
    }

    let file_data = fs::read(file_pb.clone()).unwrap();
    let gpu_processor = Arc::new(photo_editor::GpuProcessor::new(0).unwrap());
    let image_format =
        photo_editor::ImageFormat::from_ext(file_pb.extension().unwrap().to_str().unwrap())
            .unwrap();
    let mut editor =
        photo_editor::PhotoEditor::new(gpu_processor, &file_data, image_format).unwrap();

    drop(file_data);

    println!("Image loaded");

    println!("exif: {:?}", editor.exif);
    println!(
        "width: {}, height: {}",
        editor.image.width, editor.image.height
    );

    editor.set_tone(3.0, 0, 0, 0, 0, 0, None).unwrap();
    editor
        .set_brightness_tone_curve(
            None,
            Some(Array1::from_vec(vec![0, 65535])),
            Some(Array1::from_vec(vec![0, 65535])),
            None,
        )
        .unwrap();
    editor
        .set_oklch_saturation_curve(
            None,
            Some(Array1::from_vec(vec![0, 65535])),
            Some(Array1::from_vec(vec![65535, 65535])),
            None,
        )
        .unwrap();
    editor.set_lens_distortion_correction(-20).unwrap();
    editor.set_vignette(-30).unwrap();
    editor.apply_adjustments().unwrap();
    let image_data = editor
        .save(&photo_editor::image::ImageFormat::JPEG)
        .unwrap();

    file_pb.set_file_name("output.jpeg");
    fs::write(file_pb, image_data).unwrap();

    println!("Image saved");
}
