// errors.rs

use thiserror::Error;

/// photo-editorのエラー。 エラーをざっくりまとめる
#[derive(Error, Debug)]
pub enum PhotoEditorError {
    /// RAW画像
    #[error("Failed to read raw image")]
    FailedToReadRawImage {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// 標準画像
    #[error("Failed to read standard image")]
    FailedToReadStandardImage {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    // 保存
    #[error("failed to save image")]
    FailedToSaveImage {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// 非対応形式
    #[error("Unsupported read image format: {0}")]
    ReadImageUnsupportedFormat(String),

    #[error("Unsupported save image format: {0}")]
    SaveImageUnsupportedFormat(String),

    /// 補完
    #[error("Interpolation failed: {0}")]
    Interpolation(#[from] InterpolationError),

    /// GPU
    #[error("GPU compute error")]
    GpuComputeError {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("GPU initialization error")]
    GpuInitializationError {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// マスク
    #[error("Mask not found: {0}")]
    MaskNotFound(String),
}

impl PhotoEditorError {
    pub fn standard_image<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        PhotoEditorError::FailedToReadStandardImage {
            source: Some(Box::new(e)),
        }
    }

    pub fn standard_image_no_source() -> Self {
        PhotoEditorError::FailedToReadStandardImage { source: None }
    }

    pub fn raw_image<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        PhotoEditorError::FailedToReadRawImage {
            source: Some(Box::new(e)),
        }
    }

    pub fn raw_image_no_source() -> Self {
        PhotoEditorError::FailedToReadRawImage { source: None }
    }

    pub fn save<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        PhotoEditorError::FailedToSaveImage {
            source: Some(Box::new(e)),
        }
    }

    pub fn save_no_source() -> Self {
        PhotoEditorError::FailedToSaveImage { source: None }
    }

    pub fn gpu_initialization<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        PhotoEditorError::GpuInitializationError {
            source: Some(Box::new(e)),
        }
    }

    pub fn gpu_initialization_no_source() -> Self {
        PhotoEditorError::GpuInitializationError { source: None }
    }

    pub fn gpu_compute<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        PhotoEditorError::GpuComputeError {
            source: Some(Box::new(e)),
        }
    }

    pub fn gpu_compute_no_source() -> Self {
        PhotoEditorError::GpuComputeError { source: None }
    }
}

/// Errors that can occur during interpolation.
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
