using LinearAlgebraicRepresentation, ViewerGL
Lar = LinearAlgebraicRepresentation
GL = ViewerGL

filename = "/Users/paoluzzi/Documents/dev/Plasm.jl/test/svg/Lar.svg"
V,EV = Lar.svg2lar(filename)

GL.VIEW([
	GL.GLLines(V,EV,GL.COLORS[1]),
	GL.GLFrame2
]);

function pointsinout(V,EV, n=10000)
	point_in = []
	point_out = []
	point_on = []
	classify = Lar.pointInPolygonClassification(V,EV)
	for k=1:n
		queryPoint = rand(1,2)
		inOut = classify(queryPoint)
		# println("k = $k, queryPoint = $queryPoint, inOut = $inOut")
		if inOut=="p_in"
			push!(point_in, queryPoint);
		elseif inOut=="p_out"
			push!(point_out, queryPoint);
		elseif inOut=="p_on"
			push!(point_on, queryPoint);
		end
	end
	#GL.GLPoints(queryPoint,GL.COLORS[2])
	return point_in,point_out,point_on
end

#funzione parallelizzata
function pointsinout_multithreading(V,EV, n=2_000_000,t=Threads.nthreads())

  point_in = [[] for i in 1:t]
  point_out = [[] for i in 1:t]
  point_on = [[] for i in 1:t]

  classify = Lar.pointInPolygonClassification(V,EV)

  Threads.@threads for k=1:n

    #queryPoint = rand(rnglist[Threads.threadid()],1,2)
    queryPoint = rand(1,2)
    inOut = classify(queryPoint)
    # println("k = $k, queryPoint = $queryPoint, inOut = $inOut")
    if inOut=="p_in"
      push!(point_in[Threads.threadid()], queryPoint);
    elseif inOut=="p_out"
      push!(point_out[Threads.threadid()], queryPoint);
    elseif inOut=="p_on"
      push!(point_on[Threads.threadid()], queryPoint);
    end
  end
  #GL.GLPoints(queryPoint,GL.COLORS[2])
  #print(point_in,point_out,point_on)
  return vcat(point_in...),vcat(point_out...),vcat(point_on...)
end

points_in, points_out, points_on = pointsinout(V,EV);
pointsin = [vcat(points_in...) zeros(length(points_in),1)]
pointsout = [vcat(points_out...) zeros(length(points_out),1)]

polygon = [GL.GLLines(V,EV,GL.COLORS[1])];
in_mesh = [GL.GLPoints(pointsin, GL.COLORS[2])]
out_mesh = [GL.GLPoints(pointsout, GL.COLORS[3])]

result = cat([polygon,in_mesh,out_mesh])
GL.VIEW(result);
