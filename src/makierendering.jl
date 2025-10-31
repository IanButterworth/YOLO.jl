"""
    renderResult(img::Array{Float32}, predictions::Vector{YOLO.PredictLabel}, settings::YOLO.Settings; save_file::String = "")

Display image with predicted bounding boxes overlaid. Optionally save render as a file (should end in .jpg, .png etc.)
"""
function renderResult(img::Array{Float32}, predictions::Vector{YOLO.PredictLabel}, settings::YOLO.Settings; save_file::String = "")
    cols = distinguishable_colors(settings.num_classes+1, [RGB(1,1,1)])[2:end]
    img_w = settings.image_shape[1]
    img_h = settings.image_shape[2]
    img_rgb = imgslice_to_rgbimg(img)
    scene = Makie.Scene(resolution = size(img_rgb') .* 2 )
    Makie.image!(scene, img_rgb, scale_plot=false, show_axis = false, limits = Makie.FRect(0, 0, img_h, img_w))
    Makie.rotate!(scene, -0.5pi)

    for p in predictions
        alpha = clamp(p.conf,0.5,1.0)
        col = cols[p.class]
        pcol = RGBA(red(col),blue(col),green(col),alpha)
        x = p.bbox.x*img_w
        y = p.bbox.y*img_h
        w = p.bbox.w*img_w
        h = p.bbox.h*img_h
        points = [
            Point(y,x+w) => Point(y+h,x+w);
            Point(y,x) => Point(y+h,x);
            Point(y+h,x+w) => Point(y+h,x);
            Point(y,x+w) => Point(y,x);
            ]
        Makie.linesegments!(points, linewidth = 8, color=pcol)
        name = get(settings.numsdic,p.class,"")
        conf_rounded = round(p.conf, digits=2)
        Makie.text!(scene, "$name\n$(conf_rounded)",
             position = (y+3, x+3),
             align = (:left,  :top),
             color = pcol,
             textsize = 8,
             font = "Dejavu Sans"
         )
    end
    save_file != "" && Makie.save(save_file, scene)
    return scene
end

renderResult(img::KnetArray{Float32}, predictions::Vector{YOLO.PredictLabel}, settings::YOLO.Settings; save_file::String = "") =
renderResult(Array{Float32}(img), predictions, settings, save_file = save_file)
