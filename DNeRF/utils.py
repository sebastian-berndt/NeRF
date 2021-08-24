# Monkey patch logging.
def myprint(msg, *args, **kwargs):
 print(msg % args)

logging.info = myprint 
logging.warn = myprint
logging.error = myprint


def show_image(image, fmt='png'):
    image = image_utils.image_to_uint8(image)
    f = BytesIO()
    PIL.Image.fromarray(image).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


def shuffled(l):
  import random as r
  import copy
  l = copy.copy(l)
  r.shuffle(l)
  return l