// errors.rs

use thiserror::Error;
use std::io;

/// The main error type for the photo editor library.
/// It consolidates all possible errors from different modules.
#[derive(Error, Debug)]
pub enum PhotoEditorError {
    #[error("Failed to read raw image: {0}")]
    FailedToReadRawImage(#[from] ReadRawImageError),

    #[error("Failed to read standard image: {0}")]
    FailedToReadStandardImage(#[from] ReadStandardImageError),

    #[error("Interpolation failed: {0}")]
    Interpolation(#[from] InterpolationError),

    #[error("Failed to save image: {0}")]
    FailedToSaveImage(#[from] SaveImageError),

    #[error("Unsupported image format: {0}")]
    UnsupportedFormat(String),

    #[error("GPU compute error: {0}")]
    GpuComputeError(#[from] GpuComputeError),

    #[error("GPU initialization error: {0}")]
    GpuInitializationError(#[from] GpuInitializationError),
    
    #[error("Metadata error: {0}")]
    Metadata(#[from] MetadataError),

    #[error("Mask not found: {0}")]
    MaskNotFound(String),
}

/// Errors that can occur when reading a standard image format (JPEG, PNG, etc.).
#[derive(Error, Debug)]
pub enum ReadStandardImageError {
    #[error("Failed to decode image: {0}")]
    Decoding(#[from] image::ImageError),

    #[error("Failed to create image from data: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("Failed to read EXIF data: {0}")]
    Exif(#[from] exif::Error),
}

/// Errors that can occur when reading a raw image format.
#[derive(Error, Debug)]
pub enum ReadRawImageError {
    #[error("Failed to decode or process raw file: {0}")]
    Rawler(#[from] rawler::RawlerError),

    #[error("Failed to create image from data: {0}")]
    Shape(#[from] ndarray::ShapeError),
}

/// Errors that can occur during image interpolation.
#[derive(Error, Debug)]
pub enum InterpolationError {
    #[error("Input arrays must have the same length: x has {x_len}, y has {y_len}")]
    MismatchedLengths { x_len: usize, y_len: usize },

    #[error("At least two points are required for interpolation, but got {points}")]
    NotEnoughPoints { points: usize },

    #[error("x_pts must be strictly increasing, but found a non-increasing value at index {index}")]
    NotStrictlyIncreasing { index: usize },

    #[error("You must provide either a full curve or control points.")]
    MissingCurveOrControlPoints,

    #[error("The provided curve must have a length of {expected}, but it was {actual}")]
    InvalidCurveLength { expected: usize, actual: usize },

    #[error("Control points for x and y must be provided together.")]
    MissingControlPoints,

    #[error("Control point arrays cannot be empty.")]
    EmptyControlPoints,
}


/// Errors that can occur when saving an image.
#[derive(Error, Debug)]
pub enum SaveImageError {
    #[error("I/O error during save: {0}")]
    Io(#[from] io::Error),

    #[error("Encoding error during save: {0}")]
    Encoding(#[from] image::ImageError),
}

/// Errors related to initializing the GPU context (adapter, device).
#[derive(Error, Debug)]
pub enum GpuInitializationError {
    #[error("Failed to get a suitable GPU adapter.")]
    Adapter,
    #[error("Failed to get a GPU device from the adapter: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
}

/// Errors that occur during GPU compute operations.
#[derive(Error, Debug)]
pub enum GpuComputeError {
    #[error("wgpu API error: {0}")]
    Wgpu(#[from] wgpu::Error),
    
    #[error("Device poll error: {0}")]
    Poll(#[from] wgpu::PollError),

    #[error("Failed to receive data from GPU: {0}")]
    ChannelReceive(String),
    
    #[error("Asynchronous buffer mapping error: {0}")]
    BufferAsync(#[from] wgpu::BufferAsyncError),
    
    #[error("Failed to create image from GPU data: {0}")]
    Shape(#[from] ndarray::ShapeError),
}


/// Errors related to metadata processing.
#[derive(Error, Debug)]
pub enum MetadataError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("JSON serialization/deserialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Failed to read Exif data: {0}")]
    Exif(#[from] exif::Error),

    #[error("Command execution error: {0}")]
    Command(String),
}
