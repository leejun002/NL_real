import numpy as np
import scipy.io as sio
import random as rd
import math
import os


# Spectra Generator
class SpectraGenerator:
    def __init__(self):
        pass

    def pVoigt(self, pos, fw, L):
        x = np.arange(0.1, L, 1)
        sigma = fw / (2 * np.sqrt(2 * np.log(2)))
        g = 1 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-(x - pos) ** 2 / (2 * sigma ** 2))
        l = (1 / math.pi) * (fw / 2) / ((x - pos) ** 2 + (fw / 2) ** 2)
        alpha = 0.6785
        pv = (1 - alpha) * g + alpha * l
        return (pv - np.min(pv)) / (np.max(pv) - np.min(pv))

    def generate_clean_spectra(self, length, num_spectra):
        spectra = np.zeros((num_spectra, length))
        for idx in range(num_spectra):
            for _ in range(rd.randint(1, 8)):
                pos = rd.randint(5, length - 5)
                fw = rd.randint(5, 100)
                spectra[idx, :] += self.pVoigt(pos, fw, length) * rd.randint(1, 1000)
        return spectra


# Dataset Generation
def generate_raman_spectra_dataset(length, num_spectra, output_path):
    # Generate Clean Spectra
    generator = SpectraGenerator()
    clean_spectra = generator.generate_clean_spectra(length, num_spectra)

    # Generate Random Noise
    noise = np.random.normal(0, 0.1, clean_spectra.shape)

    # Combine Clean Spectra and Noise
    noisy_spectra = clean_spectra + noise

    # Save as .mat File
    dataset = {
        "clean_spectra": clean_spectra,
        "noisy_spectra": noisy_spectra,
        "noise": noise
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sio.savemat(output_path, dataset)
    print(f"Dataset saved to {output_path}")


# Main
if __name__ == "__main__":
    output_file = r'/home/jyounglee/NL/data/predict/artificial_spectra_dataset.mat'  # 원하는 .mat 파일 경로
    length = 1600  # Raman 스펙트럼 길이
    num_spectra = 100  # 생성할 스펙트럼의 수
    generate_raman_spectra_dataset(length, num_spectra, output_file)
