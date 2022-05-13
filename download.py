import sys
import os
import os.path as osp
import hashlib
import tarfile
import time
import urllib.request


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r  \\__%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))

    sys.stdout.flush()


def download(src, sha256sum, dest):
    tmp_tar = osp.join(dest, ".tmp.tar")
    try:
        urllib.request.urlretrieve(src, tmp_tar, reporthook)
    except:
        raise ConnectionError("Error: {}".format(src))

    sha256_hash = hashlib.sha256()
    with open(tmp_tar, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

        sha256_check = sha256_hash.hexdigest() == sha256sum
        print()
        print("  \\__Check sha256: {}".format("OK!" if sha256_check else "Error"))
        if not sha256_check:
            raise Exception("Error: Invalid sha256 sum: {}".format(sha256_hash.hexdigest()))

    tar_file = tarfile.open(tmp_tar, mode='r')
    tar_file.extractall(dest)
    os.remove(tmp_tar)


def main():
    """Download pre-trained SFD [1] face detector model and save under `lib/sfd/model/`.

        [1] Zhang, Shifeng, et al. "S3FD: Single shot scale-invariant face detector." Proceedings of the IEEE
            international conference on computer vision. 2017.
    """
    print("#. Download pre-trained SFD face detector model...")
    download(src='https://www.dropbox.com/s/zejv9acgk62lfbv/sfd.tar?dl=1',
             sha256sum='2bea5f1c10110e356eef3f4efd45169100b9c7704eb6e6abd309df58f34452d4',
             dest='lib')


if __name__ == '__main__':
    main()
