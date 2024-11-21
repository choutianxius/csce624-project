import json
from .mask import gen_masked
from .utils import vector_to_raster
import matplotlib.pyplot as plt


sketch = json.loads(
    '{"word":"tornado","countrycode":"US","timestamp":"2017-03-07 18:04:47.98692 UTC","recognized":true,"key_id":"5964272007905280","drawing":[[[173,160,140,86,28,9,0,5,33,151,237,254,252,226,185,70,49,38,37,50,128,199,217,222,221,209,186,121,85,67,64,69,98,158,177,180,163,133,103,87,86,91,109,151,161,164,150,130,107,89,89,94,113,147,159,159,140,119,121,130,139,141,128,124,127,132,129,134],[11,4,0,2,19,29,40,48,53,54,44,36,28,20,20,40,47,55,62,68,73,71,66,61,57,52,50,57,69,84,94,100,106,100,94,88,83,85,97,111,115,122,129,126,121,115,110,113,126,143,148,151,155,153,148,143,143,157,163,167,165,162,167,172,178,179,184,187]]]}'
)
drawing = sketch["drawing"]
masked_drawing = gen_masked(drawing)

image = vector_to_raster([drawing])[0]
image_masked = vector_to_raster([masked_drawing])[0]
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image.reshape(28, 28), cmap="gray")
ax2.imshow(image_masked.reshape(28, 28), cmap="gray")
ax1.set_title("Unmasked")
ax2.set_title("Masked")
ax1.set_axis_off()
ax2.set_axis_off()
plt.tight_layout()
plt.show()
