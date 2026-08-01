use opencv::{
    core,
    prelude::*,
};
use rand::{thread_rng, Rng};
use lazy_static::lazy_static;

lazy_static! {
    pub static ref NOISE_PALETTE: Vec<[u8; 3]> = vec![
        [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0],
        [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [128, 128, 128], [128, 0, 0], [0, 128, 0], [0, 0, 128],
        [128, 128, 0], [128, 0, 128], [0, 128, 128], [192, 192, 192]
    ];
}

const WHITE_INDEX: usize = 1;

pub fn generate_noise(height: i32, width: i32) -> Mat {
    let mut rng = thread_rng();
    let palette_len = NOISE_PALETTE.len();
    let white_idx = WHITE_INDEX;

    let mut indices = Mat::zeros(height, width, core::CV_32SC1).unwrap().to_mat().unwrap();
    for i in 0..height {
        for j in 0..width {
            let idx = rng.gen_range(0..palette_len);
            *indices.at_2d_mut::<i32>(i, j).unwrap() = idx as i32;
        }
    }

    for _ in 0..20 {
        let mut conflict = Mat::zeros(height, width, core::CV_8UC1).unwrap().to_mat().unwrap();
        for i in 0..height {
            for j in 1..width {
                let left = *indices.at_2d::<i32>(i, j).unwrap();
                let right = *indices.at_2d::<i32>(i, j-1).unwrap();
                if left == right && left != white_idx as i32 {
                    *conflict.at_2d_mut::<u8>(i, j).unwrap() = 1;
                }
            }
        }
        for i in 1..height {
            for j in 0..width {
                let up = *indices.at_2d::<i32>(i, j).unwrap();
                let down = *indices.at_2d::<i32>(i-1, j).unwrap();
                if up == down && up != white_idx as i32 {
                    *conflict.at_2d_mut::<u8>(i, j).unwrap() = 1;
                }
            }
        }

        let conflict_count = {
            let mut count = 0;
            for i in 0..height {
                for j in 0..width {
                    if *conflict.at_2d::<u8>(i, j).unwrap() == 1 {
                        count += 1;
                    }
                }
            }
            count
        };
        if conflict_count == 0 {
            break;
        }

        for i in 0..height {
            for j in 0..width {
                if *conflict.at_2d::<u8>(i, j).unwrap() == 1 {
                    let idx = rng.gen_range(0..palette_len);
                    *indices.at_2d_mut::<i32>(i, j).unwrap() = idx as i32;
                }
            }
        }
    }

    let mut noise = Mat::zeros(height, width, core::CV_8UC3).unwrap().to_mat().unwrap();
    for i in 0..height {
        for j in 0..width {
            let idx = *indices.at_2d::<i32>(i, j).unwrap() as usize;
            let color = NOISE_PALETTE[idx];
            let pixel = noise.at_2d_mut::<core::Vec3b>(i, j).unwrap();
            *pixel = core::Vec3b::from(color);
        }
    }
    noise
}
