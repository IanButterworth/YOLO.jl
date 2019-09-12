# Multithreaded control flow example for a camera image grabber, processor, saver, with UI
# Ian Butterworth 2019

# Requires julia 1.3-alpha for Threads.@spawn

using Dates

"""
    imgbuffer

Contains individual Channel buffers for the raw and processed images.
Size of Channel buffers can be set at initialization with `imgbuffer(n=1000)`
"""
struct imgbuffer
    raw::Channel{Array{Float64}}
    processed::Channel{Array{Float64}}

    function imgbuffer(;n::Int64=1000)
        raw = Channel{Array{Float64}}(n);
        processed = Channel{Array{Float64}}(n);
        return new(raw, processed)
    end
end

"""
    status

Contains individual length-1 Channel buffers for the control and information messaging between threads.
"""
struct status
    #Controls
    grab::Channel{Bool}
    #Info
    grabbed::Channel{Int64}
    processed::Channel{Int64}
    saved::Channel{Int64}

    function status()
        grab = Channel{Bool}(1); put!(grab,true)
        grabbed = Channel{Int64}(1); put!(grabbed,0)
        processed = Channel{Int64}(1); put!(processed,0)
        saved = Channel{Int64}(1); put!(saved,0)
        return new(grab, grabbed, processed, saved)
    end
end

"""
    dream(seconds)

Like sleep() except maxes out the thread for a specified number of seconds. The minimum dream time is 1
millisecond or input of `0.001`.
"""
function dream(sec::Real)
    sec ≥ 0 || throw(ArgumentError("cannot dream for $sec seconds"))
    t = Timer(sec)
    while isopen(t)
        yield()
    end
    nothing
end

"""
    img_grabber(;fps=30)

Simulates grabbing frames from a camera, blocking on waiting for new data.
"""
function img_grabber(;fps=30)
    println("img_grabber() running at $fps fps on thread: $(Threads.threadid())")
    while true
        if fetch(stat.grab) # Only grab if allowed to
            img_raw = rand(Float64, 1000, 1000);
            dream(1/fps) # Simulate grab with framerate blocking & CPU intensive delay
            put!(imgbuff.raw, img_raw)
            put!(stat.grabbed, take!(stat.grabbed) + 1) # Update stat channel
        else
            sleep(1/100) # Reduce frequency of checks on channel
        end
    end
end

"""
    img_processor(;fps=60)

Simulates processing images, blocking during processing time.
"""
function img_processor(;fps=20)
    println("img_processor() running at $fps fps on thread: $(Threads.threadid())")
    while true
        img_raw = take!(imgbuff.raw)
        img_processed = copy(img_raw);
        dream(1/fps) # Simulate processing delay with intensive CPU
        put!(imgbuff.processed, img_processed)
        put!(stat.processed, take!(stat.processed) + 1) # Update stat channel
    end
end

"""
    img_saver(;fps=10)

Simulates saving images, blocking during IO/encoding of files.
"""
function img_saver(;fps=15)
    println("img_saver() running at $fps fps on thread: $(Threads.threadid())")
    while true
        img_processed = take!(imgbuff.processed)
        dream(1/fps) # Simulate saving delay with intensive CPU
        put!(stat.saved, take!(stat.saved) + 1) # Update stat channel
    end
end

"""
    UI(;fps=1)

Basic UI, just printing a few metrics and toggling grab control every 5 seconds.
"""
function UI(;fps=1)
    println("UI() running at $fps fps on main thread: $(Threads.threadid())")
    while true
        if mod(second(now()),5) == 0 # Simulate control of grabbing in main UI (toggling every 5 seconds)
            put!(stat.grab, !take!(stat.grab)) # Toggle grab
            println("Grabbing: $(fetch(stat.grab))")
        end
        println("Grabbed: $(fetch(stat.grabbed)) (Buf $(length(imgbuff.raw.data))). Processed: $(fetch(stat.processed)) (Buf $(length(imgbuff.processed.data))). Saved: $(fetch(stat.saved))")
        sleep(1/fps) # Fix UI update rate
    end
end


const imgbuff = imgbuffer(n=1000)   # Set up image buffer channels for raw and processed images
const stat = status()               # Set up status control & info channels

Threads.@spawn img_grabber()    # Image grabber on dedicated thread
Threads.@spawn img_processor()  # Image processor on dedicated thread
Threads.@spawn img_saver()      # Image saver on dedicated thread
UI()                            # UI on main thread
