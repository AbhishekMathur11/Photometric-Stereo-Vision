def estimatePseudonormalsUncalibrated(I):


	B = None
	L = None

	U, S, V = np.linalg.svd(I, full_matrices=False)
	U3 = U[:, :3]
	S3 = np.diag(S[:3])
	V3 = V[:3, :]
	L = U3 @ S3
	B = V3

	return B, L
def plotBasRelief(B, mu, nu, lam):


    P = np.asarray([[1, 0, -mu/lam],
					[0, 1, -nu/lam],
					[0, 0,   1/lam]])
    Bp = P.dot(B)
    surface = estimateShape(Bp, s)
    plotSurface(surface, suffix=f'br_{mu}_{nu}_{lam}')

# keep all outputs visible
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'''))


# Varying parameters and plotting
parameters = [
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, -1.0, 0.0),
    (-1.0, 0.0, -1.0),
    (-1.0, -1.0, -1.0),
    (1.0, -5.0, 0.0),
    (1.0, 0.0, 5.0),
    (10.0, 1.0, 1.0)
]

for lam, mu, nu in parameters:
    plotBasRelief(B_int, mu, nu, lam)
