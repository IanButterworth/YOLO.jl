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
        col = cols[p.class]
        rect = [p.bbox.y*img_h, p.bbox.x*img_w, p.bbox.h*img_h, p.bbox.w*img_w]
        Makie.lines!(Rectangle(rect...), linewidth = 8, color=RGBA(red(col),blue(col),green(col),clamp(p.conf*1.3,0.5,1.0)))
        #Makie.poly!(scene, [Rectangle{Float32}(rect...)], color=RGBA(red(col),blue(col),green(col),clamp(p.conf*1.3,0.2,0.6)))
        name = get(settings.numsdic,p.class,"")
        conf_rounded = round(p.conf, digits=2)
        Makie.text!(scene, "$name\n$(conf_rounded)",
             position = (rect[1], rect[2]),
             align = (:left,  :top),
             colo = :black,
             textsize = 6,
             font = "Dejavu Sans"
         )
    end
    save_file != "" && Makie.save(save_file, scene)
    return scene
end

renderResult(img::KnetArray{Float32}, predictions::Vector{YOLO.PredictLabel}, settings::YOLO.Settings; save_file::String = "") =
renderResult(Array{Float32}(img), predictions, settings, save_file = save_file)
