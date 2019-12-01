"""Minimal script for reproducing the figures of the StyleGAN paper using pre-trained generators."""

import pickle

import PIL.Image
import numpy as np

import config
import dnnlib
import dnnlib.tflib as tflib

# ----------------------------------------------------------------------------
# Helpers for loading and using pre-trained generators.


synthesis_kwargs = dict(
    output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
    minibatch_size=8,
)

_Gs_cache = dict()


def load_Gs(url):
    if url not in _Gs_cache:
        with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
        _Gs_cache[url] = Gs
    return _Gs_cache[url]


# ----------------------------------------------------------------------------
# Figures 2, 3, 10, 11, 12: Multi-resolution grid of uncurated result images.


def draw_uncurated_result_figure(png, Gs, cx, cy, cw, ch, rows, lods, seed):
    print(png)
    latents = np.random.RandomState(seed).randn(
        sum(rows * 2 ** lod for lod in lods), Gs.input_shape[1]
    )
    images = Gs.run(latents, None, **synthesis_kwargs)  # [seed, y, x, rgb]

    canvas = PIL.Image.new(
        "RGB", (sum(cw // 2 ** lod for lod in lods), ch * rows), "white"
    )
    image_iter = iter(list(images))
    for col, lod in enumerate(lods):
        for row in range(rows * 2 ** lod):
            image = PIL.Image.fromarray(next(image_iter), "RGB")
            image = image.crop((cx, cy, cx + cw, cy + ch))
            image = image.resize((cw // 2 ** lod, ch // 2 ** lod), PIL.Image.ANTIALIAS)
            canvas.paste(
                image, (sum(cw // 2 ** lod for lod in lods[:col]), row * ch // 2 ** lod)
            )
    canvas.save(png)


# ----------------------------------------------------------------------------
# Figure 3: Style mixing.


def draw_style_mixing_figure(
        png, Gs, w, h, src_seeds, dst_seeds, style_ranges, **extra_kwargs
):
    print(png)
    src_latents = np.stack(
        np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds
    )
    dst_latents = np.stack(
        np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds
    )
    src_dlatents = Gs.components.mapping.run(
        src_latents, None
    )  # [seed, layer, component]
    dst_dlatents = Gs.components.mapping.run(
        dst_latents, None
    )  # [seed, layer, component]
    src_images = Gs.components.synthesis.run(
        src_dlatents, randomize_noise=False, **synthesis_kwargs, **extra_kwargs
    )[-1]
    dst_images = Gs.components.synthesis.run(
        dst_dlatents, randomize_noise=False, **synthesis_kwargs, **extra_kwargs
    )[-1]

    canvas = PIL.Image.new(
        "RGB", (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), "white"
    )
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, "RGB"), ((col + 1) * w, 0))
    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, "RGB"), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        row_images = Gs.components.synthesis.run(
            row_dlatents, randomize_noise=False, **synthesis_kwargs, **extra_kwargs
        )[-1]
        for col, image in enumerate(list(row_images)):
            canvas.paste(
                PIL.Image.fromarray(image, "RGB"), ((col + 1) * w, (row + 1) * h)
            )
    canvas.save(png)


# ----------------------------------------------------------------------------
# Figure 4: Noise detail.


def draw_noise_detail_figure(png, Gs, w, h, num_samples, seeds):
    print(png)
    canvas = PIL.Image.new("RGB", (w * 3, h * len(seeds)), "white")
    for row, seed in enumerate(seeds):
        latents = np.stack(
            [np.random.RandomState(seed).randn(Gs.input_shape[1])] * num_samples
        )
        images = Gs.run(latents, None, truncation_psi=1, **synthesis_kwargs)
        canvas.paste(PIL.Image.fromarray(images[0], "RGB"), (0, row * h))
        for i in range(4):
            crop = PIL.Image.fromarray(images[i + 1], "RGB")
            crop = crop.crop((650, 180, 906, 436))
            crop = crop.resize((w // 2, h // 2), PIL.Image.NEAREST)
            canvas.paste(crop, (w + (i % 2) * w // 2, row * h + (i // 2) * h // 2))
        diff = np.std(np.mean(images, axis=3), axis=0) * 4
        diff = np.clip(diff + 0.5, 0, 255).astype(np.uint8)
        canvas.paste(PIL.Image.fromarray(diff, "L"), (w * 2, row * h))
    canvas.save(png)


# ----------------------------------------------------------------------------
# Figure 5: Noise components.


def draw_noise_components_figure(png, Gs, w, h, seeds, noise_ranges, flips):
    print(png)
    Gsc = Gs.clone()
    noise_vars = [
        var
        for name, var in Gsc.components.synthesis.vars.items()
        if name.startswith("noise")
    ]
    noise_pairs = list(zip(noise_vars, tflib.run(noise_vars)))  # [(var, val), ...]
    latents = np.stack(
        np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds
    )
    all_images = []
    for noise_range in noise_ranges:
        tflib.set_vars(
            {
                var: val * (1 if i in noise_range else 0)
                for i, (var, val) in enumerate(noise_pairs)
            }
        )
        range_images = Gsc.run(
            latents, None, truncation_psi=1, randomize_noise=False, **synthesis_kwargs
        )
        range_images[flips, :, :] = range_images[flips, :, ::-1]
        all_images.append(list(range_images))

    canvas = PIL.Image.new("RGB", (w * 2, h * 2), "white")
    for col, col_images in enumerate(zip(*all_images)):
        canvas.paste(
            PIL.Image.fromarray(col_images[0], "RGB").crop((0, 0, w // 2, h)),
            (col * w, 0),
        )
        canvas.paste(
            PIL.Image.fromarray(col_images[1], "RGB").crop((w // 2, 0, w, h)),
            (col * w + w // 2, 0),
        )
        canvas.paste(
            PIL.Image.fromarray(col_images[2], "RGB").crop((0, 0, w // 2, h)),
            (col * w, h),
        )
        canvas.paste(
            PIL.Image.fromarray(col_images[3], "RGB").crop((w // 2, 0, w, h)),
            (col * w + w // 2, h),
        )
    canvas.save(png)


# ----------------------------------------------------------------------------
# Figure 8: Truncation trick.


def draw_truncation_trick_figure(png, Gs, w, h, seeds, psis):
    print(png)
    latents = np.stack(
        np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds
    )
    dlatents = Gs.components.mapping.run(latents, None)  # [seed, layer, component]
    dlatent_avg = Gs.get_var("dlatent_avg")  # [component]

    canvas = PIL.Image.new("RGB", (w * len(psis), h * len(seeds)), "white")
    for row, dlatent in enumerate(list(dlatents)):
        row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(
            psis, [-1, 1, 1]
        ) + dlatent_avg
        row_images = Gs.components.synthesis.run(
            row_dlatents, randomize_noise=False, **synthesis_kwargs
        )
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, "RGB"), (col * w, row * h))
    canvas.save(png)

# ----------------------------------------------------------------------------
