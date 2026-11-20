"""Run with python smoke_test.py. Fixtures and checkpoints live in a temporary directory.

Use --backend wandb to exercise the configured online logger, or --full-metrics
to include the pretrained FID and LPIPS networks (weights may be downloaded).
"""
import argparse
import json
import math
import os
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import evaluate
from data import dataloader
from models.vae import VAE, train
from utils.evaluation import compute_diversity_score, ensure_01_unit8_rgb
from utils.logger import Logger, load_checkpoint, save_checkpoint, set_seed


def tiny_G(nz=100):  # 100 -> 3x32x32
    return nn.Sequential(
        nn.ConvTranspose2d(nz, 128, 4, 1, 0, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),  # 4x4
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),   # 8x8
        nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),    # 16x16
        nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False), nn.Tanh(),                             # 32x32
    )


def tiny_D():  # 3x32x32 -> logit
    return nn.Sequential(
        nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2, True),
        nn.Conv2d(32, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
        nn.Conv2d(64, 1, 8, 1, 0),
    )


class VAEModelTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.default_generator.manual_seed(42)

    def assert_images(self, images, shape):
        self.assertEqual(tuple(images.shape), shape)
        self.assertTrue(torch.isfinite(images).all().item())
        self.assertGreaterEqual(images.min().item(), -1.0)
        self.assertLessEqual(images.max().item(), 1.0)

    @torch.no_grad()
    def test_forward_matches_encoder_and_decoder(self):
        for model, channels, latent, batch in ((VAE(), 3, 128, 2), (VAE(1, 5), 1, 5, 1)):
            with self.subTest(channels=channels, batch=batch):
                model.eval()
                images = torch.rand(batch, channels, 64, 64) * 2 - 1
                expected_mu, expected_logvar = model.encode(images)
                rng = torch.get_rng_state()
                expected_images = model.decode(model.reparameterize(expected_mu, expected_logvar))
                torch.set_rng_state(rng)
                recon, mu, logvar = model(images)
                self.assert_images(recon, tuple(images.shape))
                for actual, expected in ((mu, expected_mu), (logvar, expected_logvar)):
                    self.assertEqual(tuple(actual.shape), (batch, latent))
                    torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(recon, expected_images)

    def test_reparameterization_and_gradients(self):
        model = VAE(latent_dim=2)
        mu = torch.tensor([[2., -1.]], requires_grad=True)
        logvar = torch.tensor([[math.log(4.), math.log(0.25)]], requires_grad=True)
        rng = torch.get_rng_state()
        epsilon = torch.randn_like(mu)
        torch.set_rng_state(rng)
        z = model.reparameterize(mu, logvar)
        perturbation = torch.tensor([[2., 0.5]]) * epsilon
        torch.testing.assert_close(z, mu + perturbation)
        z.sum().backward()
        torch.testing.assert_close(mu.grad, torch.ones_like(mu))
        torch.testing.assert_close(logvar.grad, perturbation / 2)

    def test_loss_matches_hand_calculation(self):
        # Hai ảnh có SSE bằng 5 và 1. Phương sai latent lần lượt là 4 và 1.
        recon = torch.tensor([[[[1., 2.]]], [[[0., 1.]]]])
        images = torch.zeros_like(recon)
        mu = torch.tensor([[1.], [0.]])
        logvar = torch.tensor([[math.log(4.)], [0.]])
        expected_kl = (4 - math.log(4)) / 4
        for beta in (0., 0.5, 1.):
            with self.subTest(beta=beta):
                losses = VAE.loss_function(recon, images, mu, logvar, kl_weight=beta)
                expected = {"recon_loss": 3., "recon_mse": 1.5, "kl_loss": expected_kl,
                            "loss": 3 + beta * expected_kl}
                for name, value in expected.items():
                    torch.testing.assert_close(losses[name], torch.tensor(value), msg=name)

    def test_extreme_log_variance_stays_finite(self):
        model = VAE(latent_dim=2).eval()
        images = torch.zeros(1, 3, 64, 64)
        raw = torch.tensor([[-1000., 1000.]])
        with torch.no_grad():
            model.encoder.fc_logvar.weight.zero_()
            model.encoder.fc_logvar.bias.copy_(raw[0])
            mu, logvar = model.encode(images)
        bounds = torch.tensor([[model.LOGVAR_MIN, model.LOGVAR_MAX]])
        torch.testing.assert_close(logvar, bounds)
        self.assertTrue(torch.isfinite(model.reparameterize(mu, raw)).all().item())
        losses = model.loss_function(images, images, mu, raw)
        bounded = model.loss_function(images, images, mu, bounds)
        self.assertTrue(torch.isfinite(losses["loss"]).item())
        torch.testing.assert_close(losses["kl_loss"], bounded["kl_loss"])

    def test_training_step_updates_encoder_and_decoder(self):
        model = VAE(latent_dim=4).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        images = torch.rand(2, 3, 64, 64) * 2 - 1
        tracked = [model.encoder.fc_mu.weight, model.encoder.fc_logvar.weight,
                   model.decoder.fc[0].weight]
        before = [p.detach().clone() for p in tracked]
        recon, mu, logvar = model(images)
        loss = model.loss_function(recon, images, mu, logvar)["loss"]
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all().item(), name)
        optimizer.step()
        for old, param in zip(before, tracked):
            self.assertFalse(torch.equal(old, param))

    def test_inference_helpers_preserve_mode_and_batchnorm(self):
        model = VAE(latent_dim=4).eval()
        images = torch.rand(2, 3, 64, 64) * 2 - 1
        with torch.no_grad():
            mu, _ = model.encode(images)
            expected_recon = model.decode(mu)
            rng = torch.get_rng_state()
            expected_samples = model.decode(torch.randn(3, 4))
        buffers = {name: buf.clone() for name, buf in model.named_buffers()}
        for training in (True, False):
            with self.subTest(training=training):
                model.train(training)
                recon = model.reconstruct(images)
                self.assertEqual(model.training, training)
                torch.testing.assert_close(recon, expected_recon)
                torch.set_rng_state(rng)
                samples = model.sample(3)
                self.assertEqual(model.training, training)
                self.assert_images(samples, (3, 3, 64, 64))
                torch.testing.assert_close(samples, expected_samples)
                self.assertFalse(samples.requires_grad)
                self.assertFalse(recon.requires_grad)
                for name, buf in model.named_buffers():
                    torch.testing.assert_close(buf, buffers[name], msg=name)

    def test_invalid_configuration_and_input(self):
        for kwargs in ({"latent_dim": 0}, {"in_channels": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VAE(**kwargs)
        model = VAE()
        for shape in ((3, 64, 64), (1, 1, 64, 64), (1, 3, 32, 64)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape))

class VAECheckpointTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.last = self.directory / "vae_fashion_mnist_last.pt"
        self.best = self.directory / "vae_fashion_mnist_best.pt"
        self.images = torch.linspace(-1, 1, 2 * 3 * 64 * 64).reshape(2, 3, 64, 64)
        self.loader = DataLoader(TensorDataset(self.images), batch_size=2)

    def run_epoch(self, epoch, val_loss):
        # Cố định điểm validation để thử chính xác từng nhánh chọn checkpoint.
        # Model, backward, optimizer, scheduler và thao tác lưu/nạp đều chạy thật.
        def validate(model, loader, device, args):
            model.eval()
            self.reference = model.reconstruct(self.images).detach().clone()
            return val_loss, val_loss, 0.0

        argv = ["train.py", "--epochs", str(epoch), "--latent-dim", "4",
                "--batch-size", "2", "--backend", "none", "--ckpt-dir", str(self.directory)]
        if epoch > 1:
            argv.extend(["--resume", str(self.last)])
        with (
            patch.object(sys, "argv", argv),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(train, "get_dataloaders", return_value=(self.loader,) * 3),
            patch.object(train, "evaluate", side_effect=validate) as evaluate,
        ):
            train.main()
        evaluate.assert_called_once()

    def check_checkpoint(self, path, epoch, best_loss, reference):
        model = train.VAE(latent_dim=4).eval()
        info = load_checkpoint(path, models={"vae": model}, restore_rng=False)
        self.assertEqual(info["epoch"], epoch, path.name)
        self.assertEqual(info["step"], epoch, path.name)  # Một batch mỗi epoch.
        self.assertEqual(info["extra"]["best_val_loss"], best_loss, path.name)
        torch.testing.assert_close(model.reconstruct(self.images), reference, rtol=0, atol=0)

    def test_first_epoch_saves_both_checkpoints(self):
        self.run_epoch(1, 10.0)
        for path in (self.best, self.last):
            with self.subTest(checkpoint=path.name):
                self.check_checkpoint(path, 1, 10.0, self.reference)

    def test_resume_preserves_best_when_loss_worsens_or_ties(self):
        self.run_epoch(1, 10.0)
        original = self.reference
        for epoch, loss in ((2, 12.0), (3, 10.0)):
            self.run_epoch(epoch, loss)
            with self.subTest(epoch=epoch):
                self.check_checkpoint(self.best, 1, 10.0, original)
                self.check_checkpoint(self.last, epoch, 10.0, self.reference)

    def test_resume_updates_both_when_loss_improves(self):
        self.run_epoch(1, 10.0)
        self.run_epoch(2, 8.0)
        for path in (self.best, self.last):
            with self.subTest(checkpoint=path.name):
                self.check_checkpoint(path, 2, 8.0, self.reference)

    def test_resume_from_legacy_last_with_stale_best_loss(self):
        self.run_epoch(1, 10.0)
        original = self.reference
        state = torch.load(self.last, map_location="cpu", weights_only=False)
        state["extra"]["best_val_loss"] = float("inf")
        torch.save(state, self.last)
        self.run_epoch(2, 12.0)
        self.check_checkpoint(self.best, 1, 10.0, original)
        self.check_checkpoint(self.last, 2, 10.0, self.reference)


class TemporaryProjectTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        self.root = Path(temp.name)
        os.chdir(self.root)

    def create_celeba(self):
        directory = self.root / "data" / "celeba"
        directory.mkdir(parents=True)
        rows = ["image_id,partition"]
        for index, split in enumerate([0, 0, 0, 0, 1, 1, 2, 2, 2]):
            name = f"{index:06}.jpg"
            Image.new("RGB", (178, 218), (index * 25, 100, 10)).save(directory / name)
            rows.append(f"{name},{split}")
        (directory / "list_eval_partition.csv").write_text("\n".join(rows), encoding="utf-8")
        return directory


class DataTest(TemporaryProjectTest):
    def test_celeba_partition_integrity_and_cache_invalidation(self):
        directory = self.create_celeba()
        def loaders():
            return dataloader.get_dataloaders("celeba", batch_size=2, num_workers=0,
                                              strict_celeba_counts=False)

        train_dl, val_dl, test_dl = loaders()
        groups = [set(map(Path, dl.dataset.img_paths)) for dl in (train_dl, val_dl, test_dl)]
        self.assertEqual(list(map(len, groups)), [4, 2, 3])
        self.assertEqual(len(set.union(*groups)), 9)
        images, _ = next(iter(train_dl))
        self.assertEqual(tuple(images.shape), (2, 3, 64, 64))
        self.assertTrue(((images >= -1) & (images <= 1)).all().item())
        self.assertEqual([len(batch[0]) for batch in test_dl], [2, 1])
        torch.testing.assert_close(dataloader.denormalize(torch.tensor([-1., 0., 1.])),
                                   torch.tensor([0., 0.5, 1.]))
        (directory / "000000.jpg").write_bytes(b"corrupt")
        train_dl, _, _ = loaders()
        self.assertEqual(len(train_dl.dataset), 3)
        with self.assertRaises(RuntimeError):
            dataloader.get_dataloaders("celeba", batch_size=2, strict_celeba_counts=True)

    def test_fashion_mnist_split_subset_and_reproducibility(self):
        # Write valid IDX files so torchvision uses its normal offline loading path.
        raw = self.root / "data/FashionMNIST/raw"
        raw.mkdir(parents=True)
        for prefix, count in (("train", 12), ("t10k", 4)):
            pixels = np.arange(count * 28 * 28, dtype=np.int32).astype(np.uint8)
            (raw / f"{prefix}-images-idx3-ubyte").write_bytes(
                struct.pack(">4I", 2051, count, 28, 28) + pixels.tobytes())
            (raw / f"{prefix}-labels-idx1-ubyte").write_bytes(
                struct.pack(">2I", 2049, count) + bytes(i % 10 for i in range(count)))
        kwargs = dict(name="fashion_mnist", batch_size=2, val_size=3, subset=4, num_workers=0)
        tr, val, test = dataloader.get_dataloaders(**kwargs)
        self.assertEqual([len(dl.dataset) for dl in (tr, val, test)], [4, 3, 4])
        original_ids = {tr.dataset.dataset.indices[i] for i in tr.dataset.indices}
        self.assertTrue(original_ids.isdisjoint(val.dataset.indices))
        again, _, _ = dataloader.get_dataloaders(**kwargs)
        for batch, repeated in zip(tr, again):
            images, labels = batch
            self.assertEqual(tuple(images.shape), (2, 3, 64, 64))
            torch.testing.assert_close(images[:, 0], images[:, 2])
            torch.testing.assert_close(images, repeated[0])
            torch.testing.assert_close(labels, repeated[1])


class PipelineTest(TemporaryProjectTest):
    full_metrics = False

    def test_train_resume_and_evaluate(self):
        self.create_celeba()
        command = ["train.py", "--dataset", "celeba", "--data-root", "data",
                   "--batch-size", "2", "--backend", "tensorboard", "--ckpt-dir", "ckpt"]
        best = Path("ckpt/vae_celeba_best.pt")
        last = Path("ckpt/vae_celeba_last.pt")
        # Only scale the official partition cardinalities to the local fixture.
        with patch.dict(dataloader.CELEBA_EXPECTED_COUNTS, {"0": 4, "1": 2, "2": 3}):
            for epoch in (1, 2):
                args = command + ["--epochs", str(epoch)]
                if epoch == 2:
                    args.extend(["--resume", str(last)])
                with patch.object(sys, "argv", args):
                    train.main()
                info = load_checkpoint(last, {}, restore_rng=False)
                self.assertEqual((info["epoch"], info["step"]), (epoch, epoch * 2))
                self.assertTrue(math.isfinite(info["extra"]["best_val_loss"]))
            args = ["evaluate.py", "--model", "vae", "--dataset", "celeba", "--ckpt", str(best),
                    "--batch-size", "2", "--num-samples", "3"]
            if not self.full_metrics:
                args += ["--no-fid", "--no-lpips"]
            with patch.object(sys, "argv", args):
                evaluate.main()
        report = json.loads(Path("outputs/eval_vae_celeba.json").read_text(encoding="utf-8"))
        self.assertEqual((report["model"], report["num_samples"]), ("vae", 3))
        self.assertGreater(report["speed"]["image_per_sec"], 0)
        self.assertGreater(report["diversity"]["pairwise_L2"], 0)
        self.assertGreater(report["diversity"]["pixel_std"], 0)
        for value in (report["FID"], report["diversity"]["pairwise_lpips"]):
            if self.full_metrics:
                self.assertIsNotNone(value)
                self.assertTrue(math.isfinite(value))
            else:
                self.assertIsNone(value)
        self.assertTrue(any(Path("runs").rglob("events.out.tfevents.*")))
        self.assertTrue(any(Path("outputs/samples").rglob("*.png")))

    def test_metric_inputs_and_identical_images(self):
        gray = torch.tensor([[[[-1., 0., 1.]]]])
        rgb = ensure_01_unit8_rgb(gray)
        expected = torch.tensor([0, 127, 255], dtype=torch.uint8).expand(1, 3, 1, 3)
        torch.testing.assert_close(rgb, expected)
        metrics = compute_diversity_score(gray.repeat(3, 1, 1, 1), use_lpips=False)
        self.assertEqual((metrics["pixel_std"], metrics["pairwise_L2"]), (0., 0.))


class LoggerCheckpointTest(TemporaryProjectTest):
    backend = "tensorboard"
    project = "gen-models-de4"
    entity = None
    steps = 1

    def test_logging_and_multi_model_checkpoint(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(42)
        generator, discriminator = tiny_G().to(device), tiny_D().to(device)
        models = {"G": generator, "D": discriminator}
        optimizers = {k: torch.optim.Adam(m.parameters()) for k, m in models.items()}
        logger = Logger(self.backend, project=self.project, entity=self.entity, run_name="smoke-test")
        self.addCleanup(logger.finish)
        criterion = nn.BCEWithLogitsLoss()
        fixed_noise = torch.randn(4, 100, 1, 1, device=device)
        for step in range(1, self.steps + 1):
            real = torch.rand(4, 3, 32, 32, device=device) * 2 - 1
            fake = generator(torch.randn_like(fixed_noise))
            real_logits = discriminator(real)
            fake_logits = discriminator(fake.detach())
            loss_d = (criterion(real_logits, torch.ones_like(real_logits))
                      + criterion(fake_logits, torch.zeros_like(fake_logits)))
            optimizers["D"].zero_grad()
            loss_d.backward()
            optimizers["D"].step()
            logits = discriminator(fake)
            loss_g = criterion(logits, torch.ones_like(logits))
            optimizers["G"].zero_grad()
            loss_g.backward()
            optimizers["G"].step()
            self.assertTrue(torch.isfinite(loss_g + loss_d).item())
            logger.log_scalars({"loss/D": loss_d.item(), "loss/G": loss_g.item()}, step)
        generator.eval()
        with torch.no_grad():
            samples = generator(fixed_noise)
        logger.log_image_grid(samples, self.steps)
        path = "checkpoints/smoke.pt"
        save_checkpoint(path, models, optimizers, step=self.steps, extra={"noise": fixed_noise.cpu()})
        expected_rng = (random.random(), np.random.rand(), torch.rand(3))
        restored = {"G": tiny_G().to(device), "D": tiny_D().to(device)}
        opts = {k: torch.optim.Adam(m.parameters()) for k, m in restored.items()}
        info = load_checkpoint(path, restored, opts, map_location=device)
        self.assertEqual(info["step"], self.steps)
        self.assertEqual((random.random(), np.random.rand()), expected_rng[:2])
        torch.testing.assert_close(torch.rand(3), expected_rng[2])
        for key in models:
            torch.testing.assert_close(models[key].state_dict(), restored[key].state_dict())
            torch.testing.assert_close(optimizers[key].state_dict(), opts[key].state_dict())
        with torch.no_grad():
            actual = restored["G"].eval()(info["extra"]["noise"].to(device))
        torch.testing.assert_close(samples, actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["tensorboard", "wandb"], default="tensorboard")
    parser.add_argument("--entity")
    parser.add_argument("--project", default="gen-models-de4")
    parser.add_argument("--steps", type=int, default=1, help="Số bước cho smoke test generator/discriminator")
    parser.add_argument("--full-metrics", action="store_true", help="Chạy cả FID và LPIPS")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps phải >= 1")
    LoggerCheckpointTest.backend = args.backend
    LoggerCheckpointTest.entity = args.entity
    LoggerCheckpointTest.project = args.project
    LoggerCheckpointTest.steps = args.steps
    PipelineTest.full_metrics = args.full_metrics
    torch.set_num_threads(min(2, torch.get_num_threads()))
    print(f"PyTorch {torch.__version__} | Device: {('cuda' if torch.cuda.is_available() else 'cpu')}", flush=True)
    unittest.main(argv=[sys.argv[0]], verbosity=2, buffer=True)


if __name__ == "__main__":
    main()
