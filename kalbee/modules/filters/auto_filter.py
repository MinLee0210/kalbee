from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.filters.abg_filter import AlphaBetaGammaFilter
from kalbee.modules.filters.ekf_filter import ExtendedKalmanFilter


class AutoFilter:
    """
    Factory class for creating filter instances.
    """

    @staticmethod
    def from_filter(*args, mode: str = "kalmanfilter", **kwargs) -> BaseFilter:
        """
        Create a filter instance based on the specified mode.

        Args:
            mode: The type of filter to create.
                  Options:
                  - 'kalmanfilter', 'kf', 'kalman'
                  - 'extendedkalmanfilter', 'ekf'
                  - 'alphabetagamma', 'abg'
            *args: Positional arguments passed to the filter constructor.
            **kwargs: Keyword arguments passed to the filter constructor.

        Returns:
            An instance of a BaseFilter subclass.

        Raises:
            ValueError: If the mode is unknown.
        """
        mode_clean = mode.lower().replace("_", "").replace("-", "")

        if mode_clean in ["kalmanfilter", "kf", "kalman"]:
            return KalmanFilter(*args, **kwargs)

        elif mode_clean in ["extendedkalmanfilter", "ekf"]:
            return ExtendedKalmanFilter(*args, **kwargs)

        elif mode_clean in ["alphabetagamma", "abg"]:
            return AlphaBetaGammaFilter(*args, **kwargs)

        else:
            raise ValueError(
                f"Unknown filter mode: '{mode}'. Available modes: kalmanfilter, ekf, abg."
            )
