from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import math

from environments.base import Environment, State, Action
# ==================== Data Structures ====================

@dataclass
class User:
    """A user characterized by a feature vector."""
    features: np.ndarray


@dataclass
class Article(Action):
    """A news article that can be recommended."""
    id: int
    features: np.ndarray
    topic: int
    publisher: int
    published_at: float
    base_popularity: float


@dataclass
class WebsiteState(State):
    """The environment state: current catalog, current user, and time."""
    actions: List[Article]
    context: np.ndarray
    steps: int
    time_hours: float


# ==================== Environment ====================

class RecsysEnvironment(Environment):
    """
    Simulated news recommendation environment combining
    a recommender model and bandit feedback.

    Public interface:
    - __init__: Create a new environment.
    - state():   Return current state.
    - step():    Choose an article -> get click feedback.
    - sample_users(): Generate a batch of synthetic users.
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        users_per_minute: float = 30.0,
        n_user_clusters: int = 5,
        n_topics: int = 25,
        embedding_dim: int = 6,
        user_cluster_std: float = 0.30,
        topic_std: float = 0.20,
        subtopic_drift: float = 0.10,
        weight_personal: float = 5.0,
        weight_topic: float = 0.5,
        weight_publisher: float = 0.1,
        weight_article: float = 1.5,
        weight_freshness: float = 1.0,
        decay_hours: float = 36,
        global_bias: float = -2,
        score_noise: float = 0.01,
        normalize_vectors: bool = True,
        n_publishers: Optional[int] = 40,
        catalog_size: int = 6,
        new_articles_per_hour: float = 1.0,
        # NEW: fixed user pool size (defaults to 1000 as requested)
        user_pool_size: int = 1000,
    ) -> None:
        """
        Initialize the news recommendation environment.

        Args:
            rng: Random number generator for reproducibility.
            users_per_minute: Average simulated user arrivals per minute.
            n_user_clusters: Number of user clusters.
            n_topics: Number of article topics.
            embedding_dim: Dimension of feature vectors.
            user_cluster_std: Std. deviation for user features around cluster centroid.
            topic_std: Std. deviation for article features around topic centroid.
            subtopic_drift: Random drift applied to article vectors.
            weight_personal, weight_topic, weight_publisher, weight_article, weight_freshness:
                Linear weights for computing click probability.
            decay_hours: Controls freshness decay rate.
            global_bias: Constant offset in click probability computation.
            score_noise: Gaussian noise in score computation.
            normalize_vectors: Whether to normalize all embeddings.
            n_publishers: Number of publishers; auto-set if None.
            catalog_size: Number of articles in the visible catalog.
            new_articles_per_hour: Expected number of new articles appearing per hour.
            user_pool_size: Size of the fixed user pool sampled at construction.
        """

        # ---------- helpers ----------
        def _is_real_number(x) -> bool:
            return isinstance(x, (int, float)) and not isinstance(x, bool)

        def _require_real_number(name: str, x):
            if not _is_real_number(x):
                raise TypeError(f"{name} must be a real number (int or float), not {type(x).__name__}.")

        def _require_nonneg(name: str, x):
            _require_real_number(name, x)
            if x < 0:
                raise ValueError(f"{name} must be >= 0.")

        def _require_pos(name: str, x):
            _require_real_number(name, x)
            if x <= 0:
                raise ValueError(f"{name} must be > 0.")

        def _require_int(name: str, x):
            if not (isinstance(x, int) and not isinstance(x, bool)):
                raise TypeError(f"{name} must be an int (bools not allowed).")

        def _require_pos_int(name: str, x):
            _require_int(name, x)
            if x <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        # ---------- validation ----------
        if not isinstance(rng, np.random.RandomState):
            raise TypeError("rng must be an instance of np.random.RandomState.")

        _require_pos("users_per_minute", users_per_minute)

        _require_pos_int("n_user_clusters", n_user_clusters)
        _require_pos_int("n_topics", n_topics)
        _require_pos_int("embedding_dim", embedding_dim)
        _require_pos_int("catalog_size", catalog_size)

        if n_publishers is not None:
            _require_pos_int("n_publishers", n_publishers)

        _require_nonneg("new_articles_per_hour", new_articles_per_hour)
        _require_pos_int("user_pool_size", user_pool_size)  # NEW

        # float-like params must be real numbers (non-bool); no range restriction here
        for name, val in [
            ("user_cluster_std", user_cluster_std),
            ("topic_std", topic_std),
            ("subtopic_drift", subtopic_drift),
            ("weight_personal", weight_personal),
            ("weight_topic", weight_topic),
            ("weight_publisher", weight_publisher),
            ("weight_article", weight_article),
            ("weight_freshness", weight_freshness),
            ("decay_hours", decay_hours),
            ("global_bias", global_bias),
            ("score_noise", score_noise),
        ]:
            _require_real_number(name, val)

        # normalize_vectors must be a true bool
        if not isinstance(normalize_vectors, bool):
            raise TypeError("normalize_vectors must be a bool.")

        # ---------- assign core parameters ----------
        self.rng = rng
        self.users_per_minute = float(users_per_minute)
        self.embedding_dim = int(embedding_dim)
        self.n_user_clusters = int(n_user_clusters)
        self.n_topics = int(n_topics)
        self.user_cluster_std = float(user_cluster_std)
        self.topic_std = float(topic_std)
        self.subtopic_drift = float(subtopic_drift)

        self.weight_personal = float(weight_personal)
        self.weight_topic = float(weight_topic)
        self.weight_publisher = float(weight_publisher)
        self.weight_article = float(weight_article)
        self.weight_freshness = float(weight_freshness)
        self.decay_hours = float(decay_hours)
        self.global_bias = float(global_bias)
        self.score_noise = float(score_noise)
        self.normalize_vectors = bool(normalize_vectors)

        # NEW: store pool size
        self.user_pool_size = int(user_pool_size)

        # ---------- prototypes ----------
        self.topic_centroids = self._randn_normalized((self.n_topics, self.embedding_dim))
        self.user_centroids = self._randn_normalized((self.n_user_clusters, self.embedding_dim))
        self.topic_prior = np.ones(self.n_topics)
        self.user_prior = np.ones(self.n_user_clusters)

        pop_topic_raw = self.rng.lognormal(mean=0.0, sigma=0.6, size=self.n_topics)
        self.topic_popularity = self._zscore(pop_topic_raw)

        # ---------- publishers ----------
        if n_publishers is None:
            n_publishers = max(5, int(round(1.5 * self.n_topics)))
        self.publishers = list(range(int(n_publishers)))
        pop_pub_raw = self.rng.normal(loc=0.0, scale=0.4, size=int(n_publishers))
        self.publisher_popularity = {p: float(v) for p, v in zip(self.publishers, pop_pub_raw)}
        self.topic_publisher_affinity = self._topic_publisher_affinities(self.n_topics, int(n_publishers))

        # ---------- time & catalog ----------
        self.now_hours = 0.0
        self.new_articles_per_hour = float(new_articles_per_hour)

        # Fixed interval between arrivals (in minutes): since users_per_minute is arrivals/minute,
        # interval_minutes = 1 / users_per_minute
        self.fixed_interval_minutes = 1.0 / self.users_per_minute

        self._next_item_id = 0
        self.catalog: List[Article] = []
        self.catalog_size = int(catalog_size)

        max_age = math.ceil(self.catalog_size / self.new_articles_per_hour) if self.new_articles_per_hour > 0 else 36
        for _ in range(self.catalog_size):
            age = float(self.rng.uniform(0.0, int(max_age)))
            self.catalog.append(self._draw_article(self.now_hours - age))
        self.catalog.sort(key=lambda a: a.published_at, reverse=True)
        self._trim_catalog()

        # ---------- user pool & first user (NEW) ----------
        # Build a fixed pool of user contexts using sample_contexts
        self.user_pool_contexts: np.ndarray = self.sample_contexts(samples=self.user_pool_size)
        # Set the initial user by sampling from the fixed pool
        self.user = self._sample_user_from_pool()

    # =======================================================
    # Public Interface
    # =======================================================

    def state(self) -> WebsiteState:
        """Return the current catalog, user, and simulation time."""
        return WebsiteState(
            actions=list(self.catalog),
            context=self.user,
            steps=int(self.now_hours / (self.fixed_interval_minutes / 60)),
            time_hours=self.now_hours,
        )

    def step(self, article_id: int) -> bool:
        """
        Choose an article and observe a click (True/False).

        Args:
            article_id: ID of the article to recommend.

        Returns:
            bool: True if the user clicked, False otherwise.

        Raises:
            ValueError: If article_id is invalid or not in the catalog.
        """
        if not self.catalog:
            raise RuntimeError("Catalog is empty.")
        article = next((a for a in self.catalog if a.id == article_id), None)
        if article is None:
            raise ValueError(f"Article id {article_id} not in catalog.")

        p_click = np.clip(self._score(self.user, article), 0.0, 1.0)
        click = bool(self.rng.rand() < p_click)

        # Advance time deterministically
        self._advance_time(self.fixed_interval_minutes)

        # NEW: sample next user from the fixed pool for the next step
        self.user = self._sample_user_from_pool()
        return float(click)

    def sample_contexts(self, samples: int = 10_000) -> np.ndarray:
        """
        Sample a set of synthetic users for offline evaluation.

        Args:
            samples: Number of users to sample.

        Returns:
            np.ndarray: A (samples x embedding_dim) array of user feature vectors.
        """
        if samples <= 0:
            raise ValueError("samples must be positive.")
        return np.stack([self._sample_user() for _ in range(samples)])

    # =======================================================
    # Internal Simulator Logic
    # =======================================================

    def _advance_time(self, dt_minutes: float) -> None:
        if dt_minutes <= 0:
            return
        dt_hours = dt_minutes / 60.0
        self.now_hours += dt_hours

        lam = self.new_articles_per_hour * dt_hours
        n_new = self.rng.poisson(lam)
        for _ in range(n_new):
            self.catalog.insert(0, self._draw_article(self.now_hours))
        if n_new:
            self.catalog.sort(key=lambda a: a.published_at, reverse=True)
        self._trim_catalog()

    def _draw_article(self, published_at: float) -> Article:
        topic = self._categorical(self.topic_prior)
        vec = self._sample_article_vec(topic)
        pub = self._sample_publisher_for_topic(topic)
        base_pop = float(self.rng.normal(0, 1))
        art = Article(self._next_item_id, vec, topic, pub, published_at, base_pop)
        self._next_item_id += 1
        return art

    def _trim_catalog(self):
        self.catalog.sort(key=lambda a: a.published_at, reverse=True)
        self.catalog = self.catalog[: self.catalog_size]

    def _sample_user(self) -> np.ndarray:
        t = self._categorical(self.user_prior)
        c = self.user_centroids[t]
        vec = c + self.user_cluster_std * self.rng.normal(size=self.embedding_dim)
        vec = self._normalize(vec) if self.normalize_vectors else vec
        return vec

    # NEW: choose a user uniformly at random from the fixed pool
    def _sample_user_from_pool(self) -> np.ndarray:
        if self.user_pool_contexts is None or len(self.user_pool_contexts) == 0:
            # Fallback (shouldn't happen): sample fresh if pool missing
            return self._sample_user()
        idx = int(self.rng.randint(0, len(self.user_pool_contexts)))
        return self.user_pool_contexts[idx]

    def _sample_article_vec(self, topic: int) -> np.ndarray:
        c = self.topic_centroids[topic]
        noise = self.topic_std * self.rng.normal(size=self.embedding_dim)
        drift = self.subtopic_drift * self._normalize(self.rng.normal(size=self.embedding_dim))
        vec = c + noise + drift
        return self._normalize(vec) if self.normalize_vectors else vec

    def _sample_publisher_for_topic(self, topic: int) -> int:
        probs = self.topic_publisher_affinity[topic]
        return self.publishers[self._categorical(probs)]

    def _score(self, user: np.ndarray, article: Article) -> float:
        personal = float(np.dot(user, article.features))
        topic_pop = self.topic_popularity[article.topic]
        pub_pop = self.publisher_popularity[article.publisher]
        art_pop = article.base_popularity
        age = max(0.0, self.now_hours - article.published_at)
        freshness = math.exp(-age / max(1e-6, self.decay_hours))
        raw = (
            self.weight_personal * personal
            + self.weight_topic * topic_pop
            + self.weight_publisher * pub_pop
            + self.weight_article * art_pop
            + self.weight_freshness * freshness
            + self.global_bias
            + self.rng.normal(scale=self.score_noise)
        )
        return self._sigmoid(raw)

    # ==================== Math & Utils ====================

    def _randn_normalized(self, shape: Tuple[int, int]) -> np.ndarray:
        x = self.rng.normal(size=shape)
        return self._row_normalize(x)

    def _row_normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _categorical(self, probs: np.ndarray) -> int:
        probs = probs / (probs.sum() + 1e-12)
        return int(self.rng.choice(len(probs), p=probs))

    def _dirichlet(self, alpha: np.ndarray) -> np.ndarray:
        return self.rng.dirichlet(alpha)

    def _zscore(self, x: np.ndarray) -> np.ndarray:
        m, s = x.mean(), x.std()
        s = s if s > 1e-12 else 1.0
        return (x - m) / s

    def _topic_publisher_affinities(
        self, n_topics: int, n_publishers: int, focus_alpha: float = 0.3
    ) -> np.ndarray:
        affinities = np.zeros((n_topics, n_publishers))
        for t in range(n_topics):
            base = np.full(n_publishers, 1.0)
            favored = self.rng.choice(n_publishers, size=max(1, n_publishers // 4), replace=False)
            base[favored] += 3.0
            alpha = np.power(base, focus_alpha)
            alpha = np.clip(alpha, 0.05, None)
            affinities[t] = self._dirichlet(alpha)
        return affinities
