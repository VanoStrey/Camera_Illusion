mod config;
mod noise;
mod frame_utils;

use std::time::Instant;
use anyhow::{Context, Result};
use opencv::{
    core::{self, AlgorithmHint, MatTraitConst},
    highgui, imgproc, video, videoio,
    prelude::*,
};
use crate::config::*;
use crate::noise::generate_noise;
use crate::frame_utils::{quantize_frame, enforce_noise_rule};

fn main() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .context("Не удалось открыть камеру")?;

    if cam.set(videoio::CAP_PROP_FPS, 60.0).is_ok() {
        println!("Запрошены 60 fps");
    } else {
        println!("Камера не поддерживает 60 fps, останется значение по умолчанию");
    }

    let orig_w = cam.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let orig_h = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = cam.get(videoio::CAP_PROP_FPS)? as f64;
    println!("Камера: {}x{} @ {} fps", orig_w, orig_h, fps);

    let small_w = orig_w / PIXEL_BLOCK_SIZE as i32;
    let small_h = orig_h / PIXEL_BLOCK_SIZE as i32;
    let flow_w = small_w / 4;
    let flow_h = small_h / 4;

    highgui::named_window("Motion Illusion", highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    if !cam.read(&mut frame)? {
        return Err(anyhow::anyhow!("Первый кадр не прочитан"));
    }

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

    // Сетка координат для remap
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

    loop {
        let t0 = Instant::now();
        let mut frame = Mat::default();
        if !cam.read(&mut frame)? {
            break;
        }

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

        // Неподвижные области: оставляем исходную текстуру
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

        // Исправляем конфликты только в движущихся областях
        enforce_noise_rule(&mut output_small, &mag);

        let mut output_full = Mat::default();
        imgproc::resize(&output_small, &mut output_full, core::Size::new(orig_w, orig_h), 0.0, 0.0, imgproc::INTER_NEAREST)?;

        highgui::imshow("Motion Illusion", &output_full)?;
        if highgui::wait_key(1)? == 27 {
            break;
        }

        let processing_time = t0.elapsed().as_micros();
        total_processing += processing_time;
        frames_count += 1;
        if last_fps_print.elapsed().as_secs_f64() >= 1.0 {
            let avg_us = if frames_count > 0 { total_processing / frames_count as u128 } else { 0 };
            println!("FPS: {:.1} (avg {} µs/frame)", 1_000_000.0 / avg_us as f64, avg_us);
            last_fps_print = Instant::now();
            frames_count = 0;
            total_processing = 0;
        }

        prev_gray_flow = curr_gray_flow;
    }

    Ok(())
}
