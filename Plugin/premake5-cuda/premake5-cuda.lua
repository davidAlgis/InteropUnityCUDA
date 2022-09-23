if os.target() == "windows" then dofile("premake5-cuda-vs.lua") end
if os.target() == "linux" then dofile("premake5-cuda-linux.lua") end
