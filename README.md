<p align="center">
    <img src="./figs/ddpm.png" width="80%"></img>
</p>

## Realistic galaxy simulation via score-based generative models

Official code for 'Realistic galaxy simulation via score-based generative
models'.  We use a score-based generative model to produce realistic galaxy
images.  This implementation is based off of Phil Wang's <a
href="https://github.com/lucidrains/denoising-diffusion-pytorch">PyTorch
version</a> which is in turn transcribed from Jonathan Ho's official
Tensorflow version <a
href="https://github.com/hojonathanho/diffusion">here</a>. 

Above are some outputs from our model. Half of these galaxies are real and half
are generated. Check out the paper for the answer key.

<p align="center">
    <img src="./figs/shuffled_letters.png" width="80%"><img>
</p>

## Citing

If you find this work useful please consider citing our paper:

```bibtex
@article{smith2021,
    title={Realistic galaxy image simulation via score-based generative models},
    author={Michael J. Smith and James E. Geach and Ryan A. Jackson and Nikhil Arora and Connor Stone and St{\'{e}}ephane Courteau},
    year={2021}
}
```

Also be sure to check out Jonathan Ho's implementation here:

```bibtex
@article{ho2020,
    author = {{Ho}, Jonathan and {Jain}, Ajay and {Abbeel}, Pieter},
    title = "{Denoising Diffusion Probabilistic Models}",
    journal = {arXiv e-prints},
    year = 2020,
    eprint = {2006.11239},
}
```

And Jascha Sohl-Dickstein's original DDPM paper:

```bibtex
@article{sohl-dickstein2015,
    author = {{Sohl-Dickstein}, Jascha and {Weiss}, Eric A. and {Maheswaranathan}, Niru and {Ganguli}, Surya},
    title = "{Deep Unsupervised Learning using Nonequilibrium Thermodynamics}",
    journal = {arXiv e-prints},
    year = 2015,
    eprint = {1503.03585}
}
```
