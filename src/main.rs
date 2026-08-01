mod config;
mod noise;
mod frame_utils;

use std::time::Instant;
use anyhow::{Context, Result};
use opencv::{
    core::{self, AlgorithmHint, MatTraitConst},
    imgproc, video, videoio,
    prelude::*,
};
use crate::config::*;
use crate::noise::generate_noise;
use crate::frame_utils::quantize_frame;

use minifb::{Window, WindowOptions, Key};

fn main() -> Result<()> {
    // ---------- Захват видео (OpenCV) ----------
    // Пробуем 0 (основная камера) или 1, если 0 не даёт кадров
    let cam_index = 0; // поменяйте на 1, если основная камера всё ещё не работает
    let mut cam = videoio::VideoCapture::new(cam_index, videoio::CAP_AVFOUNDATION)
        .context("Не удалось открыть камеру")?;

    // Принудительно выставляем разрешение 1920x1080 @ 30 fps (веб-камеры обычно поддерживают)
    cam.set(videoio::CAP_PROP_FRAME_WIDTH, 1920.0)?;
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 1080.0)?;
    cam.set(videoio::CAP_PROP_FPS, 30.0)?;
    println!("Камера открыта, ждём первый кадр...");

    // Дожидаемся первого НЕпустого кадра (иногда первые несколько пустые)
    let mut frame = Mat::default();
    for _ in 0..300 {
        cam.read(&mut frame)?;
        if !frame.empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    if frame.empty() {
        anyhow::bail!("Не удалось получить первый кадр с камеры {}", cam_index);
    }
    println!("Камера готова, размер {}x{}", frame.cols(), frame.rows());

    let orig_w = frame.cols() as usize;
    let orig_h = frame.rows() as usize;

    // ---------- Параметры обработки ----------
    let small_w = orig_w as i32 / PIXEL_BLOCK_SIZE as i32;
    let small_h = orig_h as i32 / PIXEL_BLOCK_SIZE as i32;
    let flow_w = small_w / 4;
    let flow_h = small_h / 4;

    // ---------- Окно minifb ----------
    let mut window = Window::new(
        "Motion Illusion - minifb",
        orig_w,
        orig_h,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )
    .context("Не удалось создать окно minifb")?;

    // Буфер для minifb: каждый пиксель — u32 в формате 0RGB
    let mut display_buffer: Vec<u32> = vec![0u32; orig_w * orig_h];

    // Инициализация предыдущего кадра для оптического потока
    let mut frame_small = Mat::default();
    imgproc::resize(&frame, &mut frame_small, core::Size::new(small_w, small_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;
    let frame_q = quantize_frame(&frame_small, COLOR_LEVELS as i32);

    let mut prev_gray_flow = Mat::default();
    {
        let mut gray_small = Mat::default();
        imgproc::cvt_color(&frame_q, &mut gray_small, imgproc::COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
        imgproc::resize(&gray_small, &mut prev_gray_flow, core::Size::new(flow_w, flow_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;
    }

    let base_noise_small = generate_noise(small_h, small_w);

    let (grid_x, grid_y) = {
        let mut coords = Mat::zeros(small_h, small_w, core::CV_32FC2)?.to_mat()?;
        for y in 0..small_h {
            for x in 0..small_w {
                *coords.at_2d_mut::<core::Vec2f>(y, x).unwrap() =
                    core::Vec2f::from_array([x as f32, y as f32]);
            }
        }
        let mut parts = core::Vector::<Mat>::new();
        core::split(&coords, &mut parts)?;
        (parts.get(0)?, parts.get(1)?)
    };

    let mut last_fps_print = Instant::now();
    let mut frames_count = 0u32;
    let mut total_processing = 0u128;

    // ---------- Главный цикл ----------
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let t0 = Instant::now();

        let mut frame = Mat::default();
        if !cam.read(&mut frame)? || frame.empty() {
            // Пропускаем пустые кадры, чтобы не ломать поток
            continue;
        }

        // --- Вся прежняя обработка ---
        let mut frame_small = Mat::default();
        imgproc::resize(&frame, &mut frame_small, core::Size::new(small_w, small_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;
        let frame_q = quantize_frame(&frame_small, COLOR_LEVELS as i32);

        let mut gray_small = Mat::default();
        imgproc::cvt_color(&frame_q, &mut gray_small, imgproc::COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
        let mut curr_gray_flow = Mat::default();
        imgproc::resize(&gray_small, &mut curr_gray_flow, core::Size::new(flow_w, flow_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;

        let mut flow_flow = Mat::default();
        video::calc_optical_flow_farneback(
            &prev_gray_flow, &curr_gray_flow, &mut flow_flow,
            0.9, 1, 5, 1, 5, 1.1,
            video::OPTFLOW_FARNEBACK_GAUSSIAN,
        )?;

        let mut flow_small = Mat::default();
        imgproc::resize(&flow_flow, &mut flow_small, core::Size::new(small_w, small_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;
        let mut flow_scaled = Mat::default();
        core::multiply(&flow_small, &core::Scalar::new(4.0, 4.0, 4.0, 0.0), &mut flow_scaled, 1.0, core::CV_32F)?;

        let mut parts = core::Vector::<Mat>::new();
        core::split(&flow_scaled, &mut parts)?;
        let fx = parts.get(0)?;
        let fy = parts.get(1)?;

        let mut mag = Mat::default();
        core::magnitude(&fx, &fy, &mut mag)?;

        let mut map_x = Mat::default();
        let mut map_y = Mat::default();
        core::subtract(&grid_x, &fx, &mut map_x, &core::no_array(), core::CV_32F)?;
        core::subtract(&grid_y, &fy, &mut map_y, &core::no_array(), core::CV_32F)?;

        let mut motion_mask = Mat::default();
        core::compare(&mag, &core::Scalar::new(0.05, 0.05, 0.05, 0.0), &mut motion_mask, core::CMP_GE)?;
        let mut inv_mask = Mat::default();
        core::bitwise_not(&motion_mask, &mut inv_mask, &core::no_array())?;
        grid_x.copy_to_masked(&mut map_x, &inv_mask)?;
        grid_y.copy_to_masked(&mut map_y, &inv_mask)?;

        let mut output_small = Mat::default();
        imgproc::remap(
            &base_noise_small, &mut output_small,
            &map_x, &map_y,
            imgproc::INTER_NEAREST,
            core::BORDER_WRAP,
            core::Scalar::default(),
        )?;

        let mut output_full = Mat::default();
        imgproc::resize(&output_small, &mut output_full, core::Size::new(orig_w as i32, orig_h as i32), 0.0, 0.0, imgproc::INTER_NEAREST)?;

        // --- Перенос кадра в буфер minifb ---
        mat_to_rgb_buffer(&output_full, &mut display_buffer);

        window
            .update_with_buffer(&display_buffer, orig_w, orig_h)
            .context("Ошибка обновления окна")?;

        // Статистика FPS
        let dt = t0.elapsed().as_micros();
        total_processing += dt;
        frames_count += 1;
        if last_fps_print.elapsed().as_secs_f64() >= 1.0 {
            let avg_us = if frames_count > 0 { total_processing / frames_count as u128 } else { 0 };
            println!("FPS: {:.1} ({} µs/frame)", 1_000_000.0 / avg_us as f64, avg_us);
            last_fps_print = Instant::now();
            frames_count = 0;
            total_processing = 0;
        }

        prev_gray_flow = curr_gray_flow;
    }

    Ok(())
}

/// Копирует BGR-матрицу OpenCV в RGB-буфер для minifb (формат 0RGB).
fn mat_to_rgb_buffer(mat: &Mat, buffer: &mut [u32]) {
    let cols = mat.cols() as usize;
    let rows = mat.rows() as usize;
    let step = mat.step1(0).unwrap() as usize;
    let data = mat.data_bytes().unwrap();

    for y in 0..rows {
        for x in 0..cols {
            let offset = y * step + x * 3;
            let b = data[offset];
            let g = data[offset + 1];
            let r = data[offset + 2];
            buffer[y * cols + x] = (r as u32) << 16 | (g as u32) << 8 | b as u32;
        }
    }
}