module imgproc

export mean_subtract!, std_div!
# returns mean if all=true
function mean_subtract!(imgs; mode=:all)
    assert(mode in [:all, :pixel, :instance])

    if mode == :all
        mean_img = mean(imgs)
        imgs .-= mean_img
        return mean_img
    elseif mode == :pixel
        mean_pixels = mean(imgs, 4)
        imgs .-= mean_pixels
        return mean_pixels
     end
    imgs .-= mean(imgs, 1:length(size(imgs))-1)
end

function std_div!(imgs, all=false)
    if all
        std_img = std(imgs)
        imgs ./= std(imgs)
        return std_img
    end
    imgs ./= std(imgs, 1:length(size(imgs))-1)
end

end
