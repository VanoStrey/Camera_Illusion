pub const PIXEL_BLOCK_SIZE: u32 = 4;
#[allow(dead_code)]
pub const OUTPUT_SIZE: u32 = 640;
#[allow(dead_code)]
pub const COLOR_LEVELS: u32 = 4;
#[allow(dead_code)]
pub const MAX_DURATION_SEC: u32 = 60;
#[allow(dead_code)]
pub const MAX_FILE_SIZE: u64 = 50 * 1024 * 1024;
use opencv::{core, prelude::*};
use rand::{thread_rng, Rng};
use crate::noise::NOISE_PALETTE;

pub fn quantize_frame(frame: &Mat, levels: i32) -> Mat {
    let step = 256.0 / levels as f64;
    let mut quantized = Mat::default();
    frame.convert_to(&mut quantized, core::CV_8U, 1.0 / step, 0.0)
        .expect("quantize convert 1");
    let mut result = Mat::default();
    quantized.convert_to(&mut result, core::CV_8U, step, 0.0)
        .expect("quantize convert 2");
    result
}

#[allow(dead_code)]
pub fn preprocess_frame(frame: &Mat, x: i32, y: i32, side: i32, target_size: i32) -> Mat {
    let rect = core::Rect::new(x, y, side, side);
    let cropped = Mat::roi(frame, rect).unwrap();
    let mut result = Mat::default();
    opencv::imgproc::resize(
        &cropped, &mut result,
        core::Size::new(target_size, target_size),
        0.0, 0.0, opencv::imgproc::INTER_LINEAR,
    ).unwrap();
    result
}

#[allow(dead_code)]
pub fn shift_texture(texture: &Mat, dx: i32, dy: i32, fill_texture: &Mat) -> Mat {
    let h = texture.rows();
    let w = texture.cols();
    let mut shifted = fill_texture.clone();
    let (src_x, src_y, dst_x, dst_y, copy_w, copy_h) = if dx >= 0 && dy >= 0 {
        (0, 0, dx, dy, (w - dx).max(0), (h - dy).max(0))
    } else if dx >= 0 && dy < 0 {
        (0, -dy, dx, 0, (w - dx).max(0), (h + dy).max(0))
    } else if dx < 0 && dy >= 0 {
        (-dx, 0, 0, dy, (w + dx).max(0), (h - dy).max(0))
    } else {
        (-dx, -dy, 0, 0, (w + dx).max(0), (h + dy).max(0))
    };
    if copy_w > 0 && copy_h > 0 {
        let src_rect = core::Rect::new(src_x, src_y, copy_w, copy_h);
        let dst_rect = core::Rect::new(dst_x, dst_y, copy_w, copy_h);
        let src_roi = Mat::roi(texture, src_rect).unwrap();
        let mut dst_roi = Mat::roi_mut(&mut shifted, dst_rect).unwrap();
        src_roi.copy_to(&mut dst_roi).expect("shift copy_to");
    }
    shifted
}

/// Убирает конфликты «соседи одного цвета (кроме белого)» с вероятностью,
/// пропорциональной магнитуде движения в каждом пикселе.
/// motion_map — одноканальная матрица CV_32F того же размера, что и frame.
#[allow(dead_code)]
pub fn enforce_noise_rule(frame: &mut Mat, motion_map: &Mat) {
    // (функция остаётся без изменений, но не используется)
    let palette = &*NOISE_PALETTE;
    let white: [u8; 3] = [255, 255, 255];
    let width = frame.cols() as usize;
    let height = frame.rows() as usize;
    let mut rng = thread_rng();

    let step_frame = frame.step1(0).unwrap() as usize;
    let data = frame.data_bytes_mut().unwrap();

    let step_motion = motion_map.step1(0).unwrap() as usize;
    let motion_bytes = motion_map.data_bytes().unwrap();
    let motion_data: &[f32] = bytemuck::cast_slice(motion_bytes);

    let palette_bytes: Vec<[u8; 3]> = palette.iter().map(|c| *c).collect();
    let max_motion: f32 = 5.0;

    for _ in 0..3 {
        for y in 0..height {
            let row_start = y * step_frame;
            for x in 0..width {
                let offset = row_start + x * 3;
                let color = [data[offset], data[offset + 1], data[offset + 2]];
                if color == white {
                    continue;
                }

                let right_color = if x + 1 < width {
                    let r_off = row_start + (x + 1) * 3;
                    [data[r_off], data[r_off + 1], data[r_off + 2]]
                } else { white };

                let bottom_color = if y + 1 < height {
                    let next_row = (y + 1) * step_frame;
                    let b_off = next_row + x * 3;
                    [data[b_off], data[b_off + 1], data[b_off + 2]]
                } else { white };

                if (right_color != white && right_color == color)
                    || (bottom_color != white && bottom_color == color)
                {
                    let motion_idx = y * (step_motion / 4) + x;
                    let mag_val = motion_data[motion_idx];
                    let prob = (mag_val / max_motion).min(1.0).max(0.0);
                    if rng.gen::<f32>() < prob {
                        let mut new_color;
                        loop {
                            let idx = rng.gen_range(0..palette_bytes.len());
                            new_color = palette_bytes[idx];
                            let right_ok = right_color == white || new_color != right_color;
                            let bottom_ok = bottom_color == white || new_color != bottom_color;
                            if right_ok && bottom_ok {
                                break;
                            }
                        }
                        data[offset] = new_color[0];
                        data[offset + 1] = new_color[1];
                        data[offset + 2] = new_color[2];
                    }
                }
            }
        }
    }
}