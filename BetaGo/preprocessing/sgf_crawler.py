# coding = utf-8

import urllib
import urllib2
import os
import re
from Queue import Queue
from bs4 import BeautifulSoup

pattern = re.compile(r'index.php')
# homePage = "http://gokifu.com/index.php?p=2"
# homePage = "http://gokifu.com/kgs.php?p=805"
# homePage = "http://gokifu.com/other.php?p=310"


class Sgf_Crawler:

    def __init__(self, homepage=None):

        self.homepage = homepage if homepage is not None else "http://gokifu.com/index.php"
        self.queue = Queue()
        self.visit = set()

    def crawl_sgfs(self):
        assert self.queue.empty()
        assert len(self.visit) == 0
        self.queue.put(self.homepage)
        self.visit.add(self.homepage)
        while self.queue.qsize() != 0:
            url = self.queue.get()
            print "Current URL: ", url
            if url.find('http://', 0, 7) != 0:
                url = 'http://' + url
            req = urllib2.Request(url, headers={'User-Agent': "Magic Browser"})
            try:
                page = urllib2.urlopen(req)
            except:
                continue
            soup = BeautifulSoup(page.read(), "lxml")

            for link in soup.find_all('a'):
                address = link.get('href')
                if ".sgf" in address:
                    name = address.split('/')
                    # local_dir_path = '\\stcsuz\root\users\yimlin\sgf'
                    local_dir_path = '../../data/sgfs'
                    local_file_path = os.path.join(local_dir_path, name[-1])
                    try:
                        urllib.urlretrieve(address, local_file_path)
                    except:
                        continue
                    continue
                if "?p=" in address:
                    limit = int(address.split('?p=')[-1])
                    if limit > 310:
                        continue
                    address = "http://gokifu.com/index.php" + address
                    if address not in self.visit:
                        print address
                        self.visit.add(address)
                        self.queue.put(address)


def run_sgf_crawler():
    sgf_cralwer = Sgf_Crawler()
    sgf_cralwer.crawl_sgfs()


if __name__ == '__main__':
    run_sgf_crawler()
