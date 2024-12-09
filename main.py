import albedos
import calib_pseudonorms
import shape_estim
import sphere_render
import uncalibrated
import utils




if os.path.exists('/content/data'):
  shutil.rmtree('/content/data')

os.mkdir('/content/data')
!wget 'https://docs.google.com/uc?export=download&id=13nA1Haq6bJz0-h_7NmovvSRrRD76qiF0' -O /content/data/data.zip
!unzip "/content/data/data.zip" -d "/content/"
os.system("rm /content/data/data.zip")
data_dir = '/content/data/'


radius = 0.75 # cm
center = np.asarray([0, 0, 0]) # cm
pxSize = 7 # um
res = (3840, 2160)

light = np.asarray([1, 1, 1])/np.sqrt(3)
image = renderNDotLSphere(center, radius, light, pxSize, res)
plt.figure()
plt.imshow(image, cmap = 'gray')
plt.imsave('1b-a.png', image, cmap = 'gray')

light = np.asarray([1, -1, 1])/np.sqrt(3)
image = renderNDotLSphere(center, radius, light, pxSize, res)
plt.figure()
plt.imshow(image, cmap = 'gray')
plt.imsave('1b-b.png', image, cmap = 'gray')

light = np.asarray([-1, -1, 1])/np.sqrt(3)
image = renderNDotLSphere(center, radius, light, pxSize, res)
plt.figure()
plt.imshow(image, cmap = 'gray')
plt.imsave('1b-c.png', image, cmap = 'gray')

I, L, s = loadData(data_dir)

U, S, V = np.linalg.svd(I, full_matrices=False)
print(U.shape, S.shape, V.shape)
print(S)
B = estimatePseudonormalsCalibrated(I, L)
print(B.shape)
print(B)
albedos, normals = estimateAlbedosNormals(B)
albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
plt.imsave('1f-a.png', albedoIm, cmap = 'gray')
plt.imsave('1f-b.png', normalIm, cmap = 'rainbow')
surface = estimateShape(normals, s)
plotSurface(surface)


B, LEst = estimatePseudonormalsUncalibrated(I)
albedos, normals = estimateAlbedosNormals(B)
albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
plt.imsave('2b-a.png', albedoIm, cmap = 'gray')
plt.imsave('2b-b.png', normalIm, cmap = 'rainbow')
nx = normals[0, :].reshape(s)
ny = normals[1, :].reshape(s)
nz = normals[2, :].reshape(s)

grad_z = -nx / nz
grad_y = -ny / nz

depth_map = integrateFrankot(grad_z, grad_y)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(np.arange(s[1]), np.arange(s[0]))
ax.plot_surface(X, Y, depth_map, cmap='coolwarm', edgecolor='none')

ax.set_title("Reconstructed 3D Shape")
plt.show()
B_int = enforceIntegrability(B, s)
albedos, normals = estimateAlbedosNormals(B_int)

zx = normals[0, :] / normals[2, :]
zy = normals[1, :] / normals[2, :]
zx = zx.reshape(s)
zy = zy.reshape(s)
surface = integrateFrankot(zx, zy)

plotSurface(surface, suffix='_integrable')


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






