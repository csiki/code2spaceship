import requests
from bs4 import BeautifulSoup, SoupStrainer
import time, sys, os
from urllib.parse import urljoin
from glob import glob
from matplotlib.pyplot import imread, imsave, imshow
import matplotlib.pyplot as plt
from shutil import copyfile
import numpy as np


def dl_img(pic_url_prefix, pic_name, outdir, out_pic_name):
    pic_url = urljoin(pic_url_prefix, pic_name)
    with open(outdir + out_pic_name, 'wb') as handle:
        print(pic_url)
        s = requests.Session()
        s.max_redirects = 30
        s.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'
        response = s.get(pic_url, stream=True)

        if not response.ok:
            print(pic_url, response, file=sys.stderr)

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)


# with open(sauce, 'rt', encoding='latin1') as f:
#     soup = BeautifulSoup(f, 'html.parser')

# download all schematics from here: https://www.ex-astris-scientia.org/starship_database.htm
sauces = ['https://www.ex-astris-scientia.org/schematics/starfleet_ships1.htm',
          'https://www.ex-astris-scientia.org/schematics/starfleet_ships2.htm',
          'https://www.ex-astris-scientia.org/schematics/starfleet_ships3.htm',
          'https://www.ex-astris-scientia.org/schematics/other_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/starfleet_shuttles.htm',
          'https://www.ex-astris-scientia.org/schematics/starfleet_probes.htm',
          'https://www.ex-astris-scientia.org/schematics/starfleet_stations.htm',
          'https://www.ex-astris-scientia.org/schematics/ground_transportation1.htm',
          'https://www.ex-astris-scientia.org/schematics/prefed_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/future_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/vulcan_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/klingon_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/romulan_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/ferengi_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/cardassian_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/bajoran_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/dominion_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/borg_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/suliban_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/xindi_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/alphabeta_ships1.htm',
          'https://www.ex-astris-scientia.org/schematics/alphabeta_ships2.htm',
          'https://www.ex-astris-scientia.org/schematics/alphabeta_ships3.htm',
          'https://www.ex-astris-scientia.org/schematics/alphabeta_ships4.htm',
          'https://www.ex-astris-scientia.org/schematics/unknown_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/gamma_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/delta_ships1.htm',
          'https://www.ex-astris-scientia.org/schematics/delta_ships2.htm',
          'https://www.ex-astris-scientia.org/schematics/delta_ships3.htm',
          'https://www.ex-astris-scientia.org/schematics/delta_ships4.htm',
          'https://www.ex-astris-scientia.org/schematics/unknown_ships2.htm',
          'https://www.ex-astris-scientia.org/schematics/ancient_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/living_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/mirror_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/fake_ships.htm',
          'https://www.ex-astris-scientia.org/schematics/tas_ships1.htm',
          'https://www.ex-astris-scientia.org/schematics/tas_ships2.htm',
          'https://www.ex-astris-scientia.org/schematics/abramsverse_ships1.htm',
          'https://www.ex-astris-scientia.org/schematics/abramsverse_ships2.htm',
          'https://www.ex-astris-scientia.org/schematics/discovery_federation.htm',
          'https://www.ex-astris-scientia.org/schematics/discovery_klingon.htm',
          'https://www.ex-astris-scientia.org/schematics/discovery_alien.htm',
          'https://www.ex-astris-scientia.org/schematics/discovery_mirror.htm']

n = 0
outdir = 'data/dl/startrek/raw/'
for sauce in sauces:
    break  # TODO
    soup = BeautifulSoup(requests.get(sauce).text, 'html.parser')
    # print(soup.prettify())
    for link in soup.find_all('a', href=True):
        link = link.attrs['href']
        if link.endswith('.jpg') and not os.path.isfile(outdir + f'{n}.jpg'):
            dl_img('https://www.ex-astris-scientia.org/schematics/', link, outdir, f'{n}.jpg')
            n += 1
            time.sleep(0.1)

# filter out those without a white bg
indir = outdir
outdir = 'data/dl/startrek/white-bg-only/'
white = np.array([255, 255, 255], dtype=np.uint8)
for impath in glob(indir + '*.jpg'):
    outpath = outdir + os.path.basename(impath)
    if not os.path.isfile(outpath):
        img = imread(impath)
        if np.all(img[0, 0] > 250):
            copyfile(impath, outpath)
