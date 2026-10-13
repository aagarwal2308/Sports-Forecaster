"""One registry of every game-outcome model, so commands can loop over them."""

from .baseline import predict_coin_flip, predict_home_rate
from .elo import predict_elo

MODELS = {
    "coin_flip": predict_coin_flip,
    "home_rate": predict_home_rate,
    "elo": predict_elo,
}
