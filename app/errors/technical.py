"""Safe domain errors shared by retrieval and readiness checks."""


class TechnicalError(Exception):
    status_code = 500
    code = "technical_error"
    message = "Technical indicators could not be loaded."

    def __init__(self):
        super().__init__(self.message)


class TechnicalDataMissing(TechnicalError):
    status_code = 404
    code = "technical_data_missing"
    message = "Technical data is unavailable for this ticker."


class NoEligibleSessions(TechnicalError):
    status_code = 404
    code = "no_eligible_sessions"
    message = "No sessions before the market request date are available."


class InsufficientSessions(TechnicalError):
    status_code = 409
    code = "insufficient_sessions"
    message = "At least three eligible sessions are required."


class FeatureFilterUnavailable(TechnicalError):
    status_code = 409
    code = "feature_filter_unavailable"
    message = "Required model metadata is missing or changed during the request."


class InvalidModelMetadata(TechnicalError):
    code = "invalid_model_metadata"
    message = "Required model metadata is invalid."


class InvalidTechnicalData(TechnicalError):
    code = "invalid_technical_data"
    message = "Technical data is invalid."


class NoEligibleIndicators(TechnicalError):
    status_code = 409
    code = "no_eligible_indicators"
    message = "No selected model features exist in the technical data."
