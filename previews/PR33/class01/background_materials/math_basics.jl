### A Pluto.jl notebook ###
# v0.20.15

using Markdown
using InteractiveUtils

# ╔═╡ 9fffca86-aec9-4c26-bced-3edaa1af9a22
begin
using Pkg; Pkg.activate("..")
Pkg.instantiate()
end

# ╔═╡ 00cc1e5e-f10e-4559-b9d9-8aba44eb493e
begin
	using PlutoUI
	using Random
	using LinearAlgebra
	using HypertextLiteral
	using PlutoTeachingTools
	using ShortCodes, MarkdownLiteral
	using Random
	using Plots
	Random.seed!(8803)
end

# ╔═╡ a6bd5fd0-f856-4f1e-8e06-4e16639d436c
using Symbolics

# ╔═╡ 44c9e09f-4fb6-42b2-a265-3adedfa9b2d2
using Zygote

# ╔═╡ b129ba7c-953a-11ea-3379-17adae34924c
md"

### Math Foundations

| | | |
|-----------:|:--|:------------------|
|  Lecturer   | : | Rosemberg, Andrew |
|  Date   | : | 28 of July, 2025 |

# Background Math (_Welcome to Pluto!_)

This background material will use Pluto!
Pluto is a programming environment for _Julia_, designed to be **interactive** and **helpful**.

"

# ╔═╡ 22c65fc5-08f9-4675-89c5-20ff685c35b3
md"We should be active on the environment `LearningToControlClass/class01`"

# ╔═╡ a481b9a8-306a-4434-b5d0-e8fc7bba8482
md"Let's import some useful packages"

# ╔═╡ 4d88b926-9543-11ea-293a-1379b1b5ae64
begin
md"## 1) Calculus and linear algebra"
end

# ╔═╡ c9941b84-575c-4b98-94d2-72b59327a6a8
md" ### a) Partial Derivative

For a vector $w \in R^N$ and another vector $a \in R^N$, consider the function:

```math
f(w) = a^T w
```

"

# ╔═╡ ac439448-a766-4cb2-b76f-b4733cd14195
question_box(md"What is the partial derivative of $f$ with respect to $w_n$? What is the gradient?")

# ╔═╡ 23ff3286-be75-42b8-8327-46ebb7d8b538
md"Write you answer bellow in place of `missing`"

# ╔═╡ a33c9e20-0139-4e72-a6a1-d7e9641195cb
begin
	∂f╱∂ωₙ = missing # should be a function of w, a, n
	∇f = missing # should be a function of w, a
end

# ╔═╡ c9afb329-2aa8-48e2-8238-8f6f3977b988
begin
N = 10
n = 5
a = rand(N)
println("a=", a)
end

# ╔═╡ 529c2464-4611-405b-8ff7-b9de8158aa2d
md" **Lets check our answer (with Symbolic Differentiation):**"

# ╔═╡ ebd908a2-98c5-4abb-84cf-9a286779a736
begin
	@variables w[1:N]

	∂╱∂w = Differential(w[n])	
	f(t) = dot(a, t)

	grad_f = Symbolics.gradient(f(w), w)
	
	_ans = expand_derivatives(∂╱∂w(f(w)))
end

# ╔═╡ 570f87c7-f808-442e-b22c-d1aaec61922d
begin
	if ismissing(∂f╱∂ωₙ)
		still_missing()
	elseif ∂f╱∂ωₙ == _ans && grad_f == ∇f
		correct()
	else
		keep_working()
	end
end

# ╔═╡ aeb3a6bc-9540-11ea-0b8f-6d37412bfe68
md"### b) Gradient

Compute the gradient of the function:

```math

f_2(w_1, w_2, w_3) = w_1 w_2 + w_2 w_3 + w_3 w_1

```

"

# ╔═╡ 611c28fa-9542-11ea-1751-fbdedcfb7690
begin
	f2(w) = w[1] * w[2] + w[2] * w[3] + w[3] * w[1]
	∇f2 = (w) -> missing # replace missing with the function that computes the gradient of `f2`
end

# ╔═╡ da942073-32d5-4de5-82c0-271a0cb0e903
md" **Lets check our answer (with Automatic Differentiation - AD):**

Symbolic Differentiation may not always be possible, making it beneficial to have alternative Automatic Differentiation (AD) tools at your disposal.
"

# ╔═╡ 9972495f-79f4-4612-99db-d97c3e9d57b4
begin
	w2 = rand(3) # Random vector of size 3
	zygote_ans = f2'(w2)
end

# ╔═╡ e6948424-ce04-4bed-b9d1-ab6c5a513ffa
begin
	if ismissing(∇f2(w2))
		still_missing()
	elseif zygote_ans == ∇f2(w2)
		correct()
	else
		keep_working()
	end
end

# ╔═╡ 6f7eecec-9543-11ea-1284-dd52fce3ecca
md"
### c) Equivalence

Suppose we have the linear equation
```math

y = A v

```

"

# ╔═╡ 397ffb27-b0c4-408a-8c89-b022e6924ee0
question_box(md"Argue that $y$ can be written as a linear combination of the columns of $A$.")

# ╔═╡ f1346c3e-727a-488d-bfc8-54ffacd26987
md"### d) Vector Squared"

# ╔═╡ e0642f42-9545-11ea-14ee-fde52cb54ccc
question_box(md" Given any vector $v \in R^N$ with real entries, show that $v^T v = 0 \iff v = 0$")

# ╔═╡ 19ff8d36-9547-11ea-0e08-e5cdd8338673
md"
### e) Matrix Squared

Suppose a matrix $A = B^T B$ where matrix $B_{m,n}$ is not necessarily square ($m \neq n$).
"

# ╔═╡ ee268cb3-c7b4-4696-a8d2-54efe9ab56a7
question_box(md"Is that $A$ is square, symmetric, and PSD? Argue")

# ╔═╡ 9967004a-cd54-4b57-a813-c70634dece88
md" **Check your counter example:**
"

# ╔═╡ b8700fc7-3e81-44ac-abf2-d62bf96dcece
let
	_B = rand(5, 8) # Change here
	_A =_B'_B
	
	size(_A, 1) == size(_A,2) && # Square
	issymmetric(_A) && # Symmetric
	all(trunc.(eigen(_A).values, digits=3) .>= 0.0) && # PSD
	println("Still square, symmetric, and PSD")
end

# ╔═╡ 644e3970-4b24-4392-822b-0d11e5cbeb89
md"

#### f) Eigenvalues

Find the eigenvalues and vectors of

```math
A = 
\begin{bmatrix}
3 & 1 \\
1 & 3
\end{bmatrix}

```

Workout by hand!

Please define your eigenvalues in 😀 and your eigenvectors in 🥤

"

# ╔═╡ 0cd18898-359f-4d27-a4b9-63dc0b416981
md" **Lets check our answer:**"

# ╔═╡ eac62fea-954e-11ea-2768-39ce6f4059ab
begin

# Our Eigenvalues 
😀 = missing

# Our Eigenvectors

🥤= missing

end

# ╔═╡ 01bc0cf1-8759-49c1-b69c-edc690e23f8c
begin
	if ismissing(😀) || ismissing(🥤)
		still_missing()
	else
		checked_asnwer = eigen([ 3 1; 1 3])
		if issetequal(checked_asnwer.values, 😀) && issetequal(eachcol(round.(checked_asnwer.vectors; digits=2)), eachcol(round.(🥤; digits=2)))
			correct()
		else
			keep_working(md"😔")
		end
	end
end

# ╔═╡ f27f90c2-954f-11ea-3f93-17acb2ce4280
md"
#### f) Invertible matrix Eigenvalues

Let $A$ be an invertible matrix with eigenvectors $V$ and eigenvalues $\Lambda$
"

# ╔═╡ 0914879e-af78-4e1a-baf6-4f8fcad4d5ce
question_box(md"What are the eigenvectors (V₋₁) and eigenvalues (Λ₋₁) of $A^{−1}$?
			 
Write them as a function of $V$ and $\Lambda$")

# ╔═╡ 4eaf1b5e-7114-4b1a-bd13-6991f0c596bb
begin
	Random.seed!(8803)
	A = randn(n, n) + I * 10
	U, S, V = svd(A)
	S_new = Diagonal(max.(S, 1e-6)) # Ensure all singular values are positive
	A = U * S_new * V'
	@info A
end

# ╔═╡ 17a2fa3a-4dfc-45df-955a-068bbd1c225c
begin
	# Add your answer here
	V₋₁ = missing
	Λ₋₁ = missing
end

# ╔═╡ 1fec2e0a-c108-43a1-a4f3-557af9e215ab
let
	global V₋₁, Λ₋₁, A
	if ismissing(V₋₁) || ismissing(Λ₋₁)
		still_missing()
	else
		checked_asnwer = eigen(inv(A))
		if issetequal(checked_asnwer.values, Λ₋₁) && issetequal(eachcol(round.(checked_asnwer.vectors; digits=2)), eachcol(round.(V₋₁; digits=2)))
			correct()
		else
			keep_working(md"😔")
		end
	end
end

# ╔═╡ a6fbd265-f78a-4914-b9ec-f0596db10a4f
md"""
#### g) Least--Squares

The least-squares problem is for $A_{M\times N}$

```math
\min_{x \in \mathbb{R}^{N}} \; \|\,y - Ax\,\|_2^{2},
```
"""

# ╔═╡ ef54ff3d-f713-46c3-a758-b36166f9e898
question_box(md"How many solutions when $rank(A)=N$? Closed form?

How many solutions when $rank(A)<N$? What can you say about Null$(A)$?

How many solutions when $rank(A)=M$? What is the objective value?
")

# ╔═╡ 53b889b3-edf4-4eb2-ac4f-478b51117bf5
md"""
#### i) Regularized Least--Squares

The regularized least-squares problem is

```math
\min_{x \in \mathbb{R}^{N}} \; \|\,y - Ax\,\|_2^{2} \;+\; \delta\,\|\,x\,\|_2^{2},
```
"""

# ╔═╡ 92eda57b-a791-4507-bd30-5e81dcdf8946
question_box(md"For which $Rank(A)$ this is useful?")

# ╔═╡ 890ac713-4636-4ee3-9d3f-fb661f944576
question_box(md"Closed form solution?")

# ╔═╡ 0e3331b5-f7fc-4666-a617-d1b5f99aa162
question_box(md"""
Show that, as δ → 0, the regularized solution converges to the minimum--norm solution of the original problem:	
```math
\begin{aligned}
\min_{x \in \mathbb{R}^{N}} \;&\; \|\,x\,\|_2^{2} \\[4pt]
\text{s.t.}\;&\; A^{\mathsf T} A\,x \;=\; A^{\mathsf T} y.
\end{aligned}
```
""")

# ╔═╡ 7a0dc457-c4b4-4893-a36f-366cac98c349
begin
md"""
## 2) Interpolation

Polynomial interpolation can be unstable, even when the samples can be interpolated by a very smooth function.  Consider

```math
f(t)=\frac{1}{1+25t^{2}}.
```

Suppose that we sample this function at $M+1$ equally–spaced locations on the interval $[-1,1]$; we take  

```math
y_m = f(t_m), \qquad t_m = -1 + \frac{2m}{M}, \quad m = 0,1,\dots,M.
```
"""
end

# ╔═╡ 9afd68f1-0205-40cd-bf33-20fba8069055
begin
	f_q1(t) = 1 / (1 + 25 * t * t)
	
	t_q1(M) = [-1 + (2 * i / M) for i in range(0, M)]

	ts = -1:0.001:1
	plt_q1 = plot(ts, f_q1, xlabel="t", label="f(t) = 1 / (1 + 25 * t²)")
end

# ╔═╡ 7a49dd41-dca3-4a88-84c5-0bb00016bb8e
md"#### a) $M_{th}$ order polynomial"

# ╔═╡ f7e30484-c36c-4624-a38e-1ca0338dc735
begin
question_box(md"""
Find an $M^{\text{th}}$--order polynomial that interpolates the data for  

```math
M = 3,\,5,\,7,\,9,\,11,\,15.
```

Plot each interpolant separately, with $f(t)$ overlaid.
""")
end

# ╔═╡ 78558f34-fd3a-4a72-ae30-6966d13a8990
"""
	polynomial_fit(tm::Vector{Real}, ym::Vector{Real}, M::Int)::Function

Function to fit a polynomial of order `M` to sample points `(t,y)` using least-squares.
"""
function polynomial_fit(tm::Vector{T}, ym::Vector{T}, M::Int)::Function where {T<:Real}
	# Write your answer here in place if missing
	return (t) -> missing
end

# ╔═╡ 5db09226-55de-4fe6-b630-0d9748500579
begin
	range_M = [3, 5, 7, 9, 11, 15]
	polinomials = Vector{Function}(undef, length(range_M))
	for (i,m) in enumerate(range_M)
		tm = t_q1(m); ym = f_q1.(tm)
		polinomials[i] = polynomial_fit(tm, ym, m)
	end
end

# ╔═╡ 1a27508f-fd91-4a40-82d0-74753d3d1acd
begin
	# Plot here
end

# ╔═╡ f8ca0eaa-6fc4-40eb-9723-f8445cd145dd
function l2_distance_approx(f, g, a, b, num_points=1000)
	delta_x = (b - a) / num_points
	sum_sq_diff = 0.0
	for i in 0:(num_points-1)
		x_mid = a + (i + 0.5) * delta_x
		sum_sq_diff += (f(x_mid) - g(x_mid))^2
	end
	return sqrt(sum_sq_diff * delta_x)
end

# ╔═╡ 42210cd5-73bb-4d38-ac37-1896ed61327d
begin
	if ismissing(polinomials[1](0.0))
		still_missing()
	else
		l2_dist = l2_distance_approx.(f_q1, polinomials, -1, 1)
		_check = [0.344869;0.206622;0.145602;0.154995;0.232231;0.713119]
		if norm(_check - l2_dist) <= 1e-6
			correct()
		else
			keep_working()
		end
	end
end

# ╔═╡ 8253c6cb-ec11-473c-a24f-5be4d645b0cd
md"#### b) Cubic spline interpolation"

# ╔═╡ 590847e9-35a0-4efd-9963-7db770715192
begin
question_box(md"""
For $M+1 = 9$ equally–spaced points, find a cubic spline interpolation.
			 
Impose the endpoint derivative conditions $f'(t=\pm 1)=0$; this yields a 
$32×32$ linear system to solve (e.g. with `LinearAlgebra.inv` or `\`).
			 
Plot the resulting cubic spline with $f(t)$ overlaid.
""")
end

# ╔═╡ 3b4e9c20-44fd-4bcc-ab61-231cab22f6ff
# Implement and replace here 
cobic_spline = (t) -> missing

# ╔═╡ 8155e2d3-5a82-42d4-8450-297f88da190a
begin
	# Plot here
end

# ╔═╡ 002cc5c1-094e-4133-9108-451ff74e8f2f
let
	global cobic_spline
	if ismissing(cobic_spline(0.0))
		still_missing()
	else
		l2_dist = l2_distance_approx(f_q1, cobic_spline, -1, 1)
		if l2_dist < 0.145602
			correct()
		else
			keep_working()
		end
	end
end

# ╔═╡ 33a6337a-2d74-49a7-bc3a-8f92b78ff16d
md"#### c) Quadratic B-spline Superposition
Suppose that $$f(t)$$ is a second-order spline formed from five shifted quadratic B--splines:

```math
f(t)=\sum_{k=0}^{4}\alpha_k\, b_2(t-k),
```

where the basis function $$b_2(t)$$ is

```math
b_2(t)=
\begin{cases}
\dfrac{(t+\tfrac32)^2}{2}, & -\tfrac32\le t\le -\tfrac12,\\[6pt]
-t^{2}+\dfrac34,           & -\tfrac12\le t\le \tfrac12,\\[6pt]
\dfrac{(t-\tfrac32)^2}{2}, &  \tfrac12\le t\le \tfrac32,\\[6pt]
0,                         & |t|\ge \tfrac32.
\end{cases}
```

"

# ╔═╡ 9a170bfa-5ce5-46a5-9024-16394f51289b
begin
question_box(md"""

Find the corresponding coefficients $$\alpha_k$$ so that:

```math
f(0)=-1,\quad f(1)=-1,\quad f(2)=2,\quad f(3)=5,\quad f(4)=1.
```
""")
end

# ╔═╡ 6b006a1e-e53c-490e-ab61-772a168f0064
begin
	# Write your answer here
	alpha = missing # This should a vector of floats
end

# ╔═╡ 4ffc2c46-7c78-4afa-9ea3-888b1790b291
let
	global alpha
	if ismissing(alpha)
		still_missing()
	else
		_check = [-1.08975469, -1.46147186,  1.85858586,  6.30995671,  0.28167388]
		if norm(_check - alpha) <= 1e-6
			correct()
		else
			keep_working()
		end
	end
end

# ╔═╡ ad73cc63-2872-431f-803d-8a986993fce2
Columns(question_box(md"""
Now let  

```math
f(t)=\sum_{n=0}^{N-1}\alpha_n\, b_2(t-n)
```
be a superposition of $$N$$ quadratic B--splines.
			 
Describe how to construct the $$N\times N$$ matrix $$A$$ that maps the coefficient vector $$\boldsymbol{\alpha}$$ to the sample vector

```math
\begin{bmatrix}
f(0)\\
f(1)\\
\vdots\\
f(N-1)
\end{bmatrix}
=
A
\begin{bmatrix}
\alpha_0\\
\alpha_1\\
\vdots\\
\alpha_{N-1}
\end{bmatrix}.
```

And Show that the matrix $$A$$ is invertible for every $$N$$.
"""),
tip(md"""
 $$A$$ is invertible iff $$A\mathbf{x}=0$$ only when $$\mathbf{x}=0$$.  
Write $$A$$ as $$I+G$$ for a matrix $$G$$ you can evaluate, and show  

```math
\|A\mathbf{x}\|_2=\|\mathbf{x}+G\mathbf{x}\|_2\ge \|\mathbf{x}\|_2-\|G\mathbf{x}\|_2.
```

Demonstrate that $$|G\mathbf{x}|_2<|\mathbf{x}|_2$$ by proving $$G\mathbf{x}$$ is the sum of two vectors whose norms are each strictly less than $$|\mathbf{x}|_2/2$$.
""")
)

# ╔═╡ 4634c856-9553-11ea-008d-3539195970ea
md"## Final notes"

# ╔═╡ 4d0ebb46-9553-11ea-3431-2d203f594815
md"If anything about this introduction left you confused, something doesn't work, or you have a cool new idea -- don't hesitate to contact me!"

# ╔═╡ d736e096-9553-11ea-3ba5-277afde1afe7
html"""<p>We're almost done! Scroll to the top of the notebook, and click on <img src="https://cdn.jsdelivr.net/gh/ionic-team/ionicons@5.5.1/src/svg/share-outline.svg" style="width: 1em; height: 1em; margin-bottom: -.2em;"> to see the export options - or you can always share this notebook's save file. (The file is pure Julia, by the way, and it's runnable! and git friendly)</p>"""

# ╔═╡ 1deaaf36-9554-11ea-3dae-85851f73dbc6
md"
Check out the amzing packages we used here:

 - [Pluto.jl](https://github.com/fonsp/Pluto.jl) : Iteractive and light notebooks;
 - [Distributions.jl](https://github.com/JuliaStats/Distributions.jl): Easy handling of many distributions with nice type system;
 - [Plots.jl](https://github.com/JuliaPlots/Plots.jl): Julia base Plotting package that accepts many backagens (even from other programming languages);
 - [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) : Symmbolic Algebra;
 - [Zygote.jl](https://github.com/FluxML/Zygote.jl): AD

And others that are useful for Machine Learning:
 - [JuMP.jl](https://github.com/jump-dev/JuMP.jl): Amazing Optimization package that interfaces with most solvers;
 - [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl): A Machine Learning Framework for Julia.
"

# ╔═╡ Cell order:
# ╟─b129ba7c-953a-11ea-3379-17adae34924c
# ╟─22c65fc5-08f9-4675-89c5-20ff685c35b3
# ╟─9fffca86-aec9-4c26-bced-3edaa1af9a22
# ╟─a481b9a8-306a-4434-b5d0-e8fc7bba8482
# ╠═00cc1e5e-f10e-4559-b9d9-8aba44eb493e
# ╟─4d88b926-9543-11ea-293a-1379b1b5ae64
# ╟─c9941b84-575c-4b98-94d2-72b59327a6a8
# ╟─ac439448-a766-4cb2-b76f-b4733cd14195
# ╟─23ff3286-be75-42b8-8327-46ebb7d8b538
# ╠═a33c9e20-0139-4e72-a6a1-d7e9641195cb
# ╠═c9afb329-2aa8-48e2-8238-8f6f3977b988
# ╟─529c2464-4611-405b-8ff7-b9de8158aa2d
# ╠═a6bd5fd0-f856-4f1e-8e06-4e16639d436c
# ╠═ebd908a2-98c5-4abb-84cf-9a286779a736
# ╟─570f87c7-f808-442e-b22c-d1aaec61922d
# ╟─aeb3a6bc-9540-11ea-0b8f-6d37412bfe68
# ╠═611c28fa-9542-11ea-1751-fbdedcfb7690
# ╟─da942073-32d5-4de5-82c0-271a0cb0e903
# ╠═44c9e09f-4fb6-42b2-a265-3adedfa9b2d2
# ╠═9972495f-79f4-4612-99db-d97c3e9d57b4
# ╟─e6948424-ce04-4bed-b9d1-ab6c5a513ffa
# ╟─6f7eecec-9543-11ea-1284-dd52fce3ecca
# ╟─397ffb27-b0c4-408a-8c89-b022e6924ee0
# ╟─f1346c3e-727a-488d-bfc8-54ffacd26987
# ╟─e0642f42-9545-11ea-14ee-fde52cb54ccc
# ╟─19ff8d36-9547-11ea-0e08-e5cdd8338673
# ╟─ee268cb3-c7b4-4696-a8d2-54efe9ab56a7
# ╟─9967004a-cd54-4b57-a813-c70634dece88
# ╠═b8700fc7-3e81-44ac-abf2-d62bf96dcece
# ╟─644e3970-4b24-4392-822b-0d11e5cbeb89
# ╟─0cd18898-359f-4d27-a4b9-63dc0b416981
# ╠═eac62fea-954e-11ea-2768-39ce6f4059ab
# ╟─01bc0cf1-8759-49c1-b69c-edc690e23f8c
# ╟─f27f90c2-954f-11ea-3f93-17acb2ce4280
# ╟─0914879e-af78-4e1a-baf6-4f8fcad4d5ce
# ╠═4eaf1b5e-7114-4b1a-bd13-6991f0c596bb
# ╠═17a2fa3a-4dfc-45df-955a-068bbd1c225c
# ╟─1fec2e0a-c108-43a1-a4f3-557af9e215ab
# ╟─a6fbd265-f78a-4914-b9ec-f0596db10a4f
# ╟─ef54ff3d-f713-46c3-a758-b36166f9e898
# ╟─53b889b3-edf4-4eb2-ac4f-478b51117bf5
# ╟─92eda57b-a791-4507-bd30-5e81dcdf8946
# ╟─890ac713-4636-4ee3-9d3f-fb661f944576
# ╟─0e3331b5-f7fc-4666-a617-d1b5f99aa162
# ╟─7a0dc457-c4b4-4893-a36f-366cac98c349
# ╠═9afd68f1-0205-40cd-bf33-20fba8069055
# ╟─7a49dd41-dca3-4a88-84c5-0bb00016bb8e
# ╟─f7e30484-c36c-4624-a38e-1ca0338dc735
# ╠═78558f34-fd3a-4a72-ae30-6966d13a8990
# ╠═5db09226-55de-4fe6-b630-0d9748500579
# ╠═1a27508f-fd91-4a40-82d0-74753d3d1acd
# ╟─f8ca0eaa-6fc4-40eb-9723-f8445cd145dd
# ╟─42210cd5-73bb-4d38-ac37-1896ed61327d
# ╟─8253c6cb-ec11-473c-a24f-5be4d645b0cd
# ╟─590847e9-35a0-4efd-9963-7db770715192
# ╠═3b4e9c20-44fd-4bcc-ab61-231cab22f6ff
# ╠═8155e2d3-5a82-42d4-8450-297f88da190a
# ╟─002cc5c1-094e-4133-9108-451ff74e8f2f
# ╟─33a6337a-2d74-49a7-bc3a-8f92b78ff16d
# ╟─9a170bfa-5ce5-46a5-9024-16394f51289b
# ╠═6b006a1e-e53c-490e-ab61-772a168f0064
# ╟─4ffc2c46-7c78-4afa-9ea3-888b1790b291
# ╟─ad73cc63-2872-431f-803d-8a986993fce2
# ╟─4634c856-9553-11ea-008d-3539195970ea
# ╟─4d0ebb46-9553-11ea-3431-2d203f594815
# ╟─d736e096-9553-11ea-3ba5-277afde1afe7
# ╟─1deaaf36-9554-11ea-3dae-85851f73dbc6
