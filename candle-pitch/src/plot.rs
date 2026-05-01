use std::path::Path;

use kuva::plot::LinePlot;
use kuva::render::layout::Layout;
use kuva::render::plots::Plot;

use candle_pitch::notes::Note;

type DynError = Box<dyn std::error::Error>;

const PLOT_WIDTH: f64 = 800.0;
const PLOT_HEIGHT: f64 = 600.0;
const NOTE_COLOR: &str = "steelblue";
const Y_AXIS_MIN_HZ: f64 = 50.0;
const Y_AXIS_MAX_HZ: f64 = 2000.0;

pub fn render_notes(notes: &[Note], path: &Path) -> Result<(), DynError> {
    let plots: Vec<Plot> = notes
        .iter()
        .map(|note| {
            let data: Vec<(f64, f64)> = note
                .points
                .iter()
                .map(|p| (p.time as f64, p.frequency as f64))
                .collect();
            LinePlot::new()
                .with_data(data)
                .with_color(NOTE_COLOR)
                .with_stroke_width(1.5)
                .into()
        })
        .collect();

    let layout = Layout::auto_from_plots(&plots)
        .with_width(PLOT_WIDTH)
        .with_height(PLOT_HEIGHT)
        .with_x_label("time (s)")
        .with_y_label("frequency (Hz)")
        .with_log_y()
        .with_y_axis_min(Y_AXIS_MIN_HZ)
        .with_y_axis_max(Y_AXIS_MAX_HZ);

    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    match extension {
        "png" => {
            let png = kuva::render_to_png(plots, layout, 2.0)
                .map_err(|e| format!("png render failed: {e}"))?;
            std::fs::write(path, png)?;
        }
        "svg" => {
            let svg = kuva::render_to_svg(plots, layout);
            std::fs::write(path, svg)?;
        }
        _ => return Err(format!("plot path must end in .png or .svg: {}", path.display()).into()),
    }
    Ok(())
}
