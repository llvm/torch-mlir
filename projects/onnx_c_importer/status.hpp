#pragma once

#include <optional>

/// A light-weight status. It only encapsulates success/failure.
/// Full error information will be set on the ModelInfo.
class Status {
public:
  static Status success(bool isSuccess = true) { return Status(isSuccess); }
  static Status failure(bool isFailure = true) { return Status(!isFailure); }

  bool is_success() { return is_success_; }

private:
  Status(bool is_success) : is_success_(is_success) {}
  bool is_success_;
};

static inline bool succeeded(Status status) { return status.is_success(); }
static inline bool failed(Status status) { return !status.is_success(); }

// (inspired by std::nullopt_t)
struct FailureT {
  enum class _Construct { _Token };

  explicit constexpr FailureT(_Construct) noexcept {}

  operator Status() const { return Status::failure(); }
};

inline constexpr FailureT failure{FailureT::_Construct::_Token};

// (inspired by std::nullopt_t)
struct SuccessT {
  enum class _Construct { _Token };

  explicit constexpr SuccessT(_Construct) noexcept {}

  operator Status() const { return Status::success(); }
};

inline constexpr SuccessT success{SuccessT::_Construct::_Token};

// (see llvm::FailureOr)
template <typename T> class [[nodiscard]] FailureOr : public std::optional<T> {
public:
  FailureOr(FailureT) : std::optional<T>() {}
  FailureOr() : FailureOr(failure) {}
  FailureOr(T &&Y) : std::optional<T>(std::forward<T>(Y)) {}
  FailureOr(const T &Y) : std::optional<T>(Y) {}
  template <typename U,
            std::enable_if_t<std::is_constructible<T, U>::value> * = nullptr>
  FailureOr(const FailureOr<U> &Other)
      : std::optional<T>(failed(Other) ? std::optional<T>()
                                       : std::optional<T>(*Other)) {}

  operator Status() const { return Status::success(has_value()); }

private:
  /// Hide the bool conversion as it easily creates confusion.
  using std::optional<T>::operator bool;
  using std::optional<T>::has_value;
};
