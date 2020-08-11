//===- device.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

// Structured similarly to code from git@github.com:pytorch/xla.git

#pragma once

#include <iostream>
#include <string>

namespace torch_mlir {

enum class DeviceType { CPU, MLIR };

/// Model a pytorch device, which determines the location of a buffer in
/// pytorch.
struct Device {
  Device() = default;
  explicit Device(const std::string &device_spec);
  Device(DeviceType hw_type, int ordinal)
      : hw_type(hw_type), ordinal(ordinal) {}

  bool operator==(const Device &other) const { return compare(other) == 0; }

  bool operator!=(const Device &other) const { return compare(other) != 0; }

  bool operator<(const Device &rhs) const { return compare(rhs) < 0; }

  int compare(const Device &rhs) const {
    if (hw_type != rhs.hw_type) {
      return hw_type < rhs.hw_type ? -1 : +1;
    }
    return ordinal < rhs.ordinal ? -1 : (ordinal > rhs.ordinal ? +1 : 0);
  }

  std::string ToString() const;

  friend std::ostream &operator<<(std::ostream &os, const Device &device) {
    os << device.ToString();
    return os;
  }

  size_t hash() const { return std::hash<std::string>{}(ToString()); }

  DeviceType hw_type = DeviceType::CPU;
  int ordinal = 0;
};

const Device *GetDefaultDevice();

static inline const Device &GetDeviceOrDefault(const Device *device) {
  return device != nullptr ? *device : *GetDefaultDevice();
}

} // namespace torch_mlir
