module imageGenerationWithDiffusionModels

using MAT
using Images

function load_digits_data(filepath::String)
    matfile = matread(filepath) 
    return matfile  
end

end

