from scipy import stats


def test_normal_pdf():
    """Test to reassure."""
    mu = 5
    sigma = 3
    low = 0
    high = 15

    a, b = (low - mu) / sigma, (high - mu) / sigma
    x = 3.1223

    result = stats.truncnorm.pdf(x, a=a, b=b, loc=mu, scale=sigma)

    norm_x = stats.norm.pdf(x, loc=mu, scale=sigma)
    norm_high = stats.norm.cdf(high, loc=mu, scale=sigma)
    norm_low = stats.norm.cdf(low, loc=mu, scale=sigma)

    assert result == norm_x / (norm_high - norm_low)
