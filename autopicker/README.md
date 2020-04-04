This repository is for an autopicking method that is still in its very early stages. There has been a lot of work put into finding a way to automatically contour layers in ice and snow radar data, with some moderate success for picking the surface and bed in a radargram. However, there has been much less success in picking internal layers, and certainly not the tens to hundreds of layers that can exist in a single line at the same time. This is important because picking is a time-consuming process, and is often limited by how the human eye can identify contours. Instead, automating this process so that we can achieve an objectively picked set of contours that does not then need additional human validation is prudent.


Most of the previous algorithms have relied heavily on domain knowledge within radioglaciology, but one route that ImpDAR is exploring for an autopicker is taking an image processing approach. We can take advantage of scikit-image's [find_contours())[https://scikit-image.org/docs/0.8.0/api/skimage.measure.find_contours.html] method that implements the Marching Squares algorithm, and can help us find contours in glaciers and ice sheets.

If you would like to read more about the efforts to create an autopicker within the radioglaciological community, here are a list of references that we have found helpful:
Hale, D.,v2009: Structure-oriented smoothing and semblance. CWP Report 635, 10 pp.,
http://inside.mines.edu/~dhale/papers/Hale09StructureOrientedSmoothingAndSemblance.pdf

Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart, Tony Yu and the scikit-image contributors. scikit-image: Image processing in Python, PeerJ 2:e453 (2014)

Sheriff, R. E., and L. P. Geldart, 1995: Exploration Seismology, Second Edition. Cambridge University Press, 592 pp.

Lorensen, William and Harvey E. Cline. Marching Cubes: A High Resolution 3D Surface Construction Algorithm. Computer Graphics (SIGGRAPH 87 Proceedings) 21(4) July 1987, p. 163-170).

Ferro, A., & Bruzzone, L. (2012). Automatic extraction and analysis of ice layering in radar sounder data. IEEE Transactions on Geoscience and Remote Sensing, 51(3), 1622-1634.

Kamangir, H., Rahnemoonfar, M., Dobbs, D., Paden, J., & Fox, G. (2018, July). Deep hybrid wavelet network for ice boundary detection in radra imagery. In IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium (pp. 3449-3452). IEEE.

Crandall, D. J., Fox, G. C., & Paden, J. D. (2012, November). Layer-finding in radar echograms using probabilistic graphical models. In Proceedings of the 21st International Conference on Pattern Recognition (ICPR2012) (pp. 1530-1533). IEEE.

Mitchell, J. E., Crandall, D. J., Fox, G. C., & Paden, J. D. (2013, July). A semi-automatic approach for estimating near surface internal layers from snow radar imagery. In 2013 IEEE International Geoscience and Remote Sensing Symposium-IGARSS (pp. 4110-4113). IEEE.

Xu, M., Fan, C., Paden, J. D., Fox, G. C., & Crandall, D. J. (2018, March). Multi-task spatiotemporal neural networks for structured surface reconstruction. In 2018 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 1273-1282). IEEE.
