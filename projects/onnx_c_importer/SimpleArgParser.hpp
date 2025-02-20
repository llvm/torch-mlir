//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

// Minimal argument parser to avoid depending on LLVM support lib.

#pragma once

#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

namespace torch_mlir_onnx {

struct positional_tag {
  constexpr static bool value = true;
};
struct optional_tag {
  constexpr static bool value = false;
};

class base_arg {
public:
  virtual bool Parse(std::span<char *>::iterator &args_it,
                     const std::span<char *>::iterator end) = 0;
  virtual ~base_arg() {};
  virtual void PrintUsage() const = 0;
  virtual void PrintDesc() const = 0;
};

struct args {
private:
  inline static std::vector<base_arg *> positionals_;
  inline static std::unordered_map<std::string_view, base_arg *> optionals_;

public:
  static void RegisterPositional(base_arg *arg) { positionals_.push_back(arg); }
  static void RegisterOptional(std::string_view name, base_arg *arg) {
    optionals_.emplace(name, arg);
  }
  static bool ParseArgs(std::span<char *>::iterator args_it,
                        const std::span<char *>::iterator end,
                        const std::string &command_desc = "") {

    if (args_it == end) {
      PrintHelp(command_desc);
      return false;
    }

    for (; args_it != end && (*args_it)[0] == '-';
         args_it = std::next(args_it)) {
      if (optionals_.count(std::string_view(*args_it))) {
        if (!(optionals_[std::string_view(*args_it)])->Parse(args_it, end))
          return false;
      } else {
        std::cerr << "Unknown argument: " << *args_it << "\n";
        return false;
      }
    }

    for (auto &p : positionals_) {
      if (args_it == end) {
        std::cerr << "Positional argument missing:";
        p->PrintUsage();
        std::cerr << "\n";
        return false;
      }

      if (!p->Parse(args_it, end))
        return false;

      args_it = std::next(args_it);
    }

    if (args_it != end) {
      std::cerr << "Unexpected positional argument: " << *args_it << "\n";
      return false;
    }

    return true;
  }

  static void PrintHelp(const std::string &command_desc) {
    std::cerr << "usage: torch-mlir-import-onnx";

    for (auto &o : optionals_) {
      o.second->PrintUsage();
    }

    for (auto &p : positionals_) {
      p->PrintUsage();
    }

    std::cerr << "\n";
    if (command_desc != "") {
      std::cerr << "\n";
      std::cerr << command_desc << "\n";
    }
    std::cerr << "\n";
    std::cerr << "positional arguments: \n";
    for (auto &p : positionals_) {
      p->PrintDesc();
    }
    std::cerr << "\n";
    std::cerr << "optional arguments: \n";
    for (auto &o : optionals_) {
      o.second->PrintDesc();
    }
  }
};

template <typename P, typename T> class arg : public base_arg {

private:
  std::string desc_;
  std::string name_;
  T value_;
  std::string value_desc_;

  void register_arg() {
    if constexpr (P::value) {
      args::RegisterPositional(this);
    } else {
      args::RegisterOptional(name_, this);
    }
  }

  virtual bool Parse(std::span<char *>::iterator &args_it,
                     const std::span<char *>::iterator end) override {
    if (args_it == end)
      return false;
    if constexpr (!P::value) {
      assert(std::string_view(*args_it) == std::string_view(name_));
      if constexpr (!std::is_same<bool, T>::value)
        args_it = std::next(args_it);
      if (args_it == end)
        return false;
    }

    if constexpr (std::is_same<std::string, T>::value) {
      value_ = std::string(*args_it);
      return true;
    } else if constexpr (std::is_same<bool, T>::value) {
      value_ = true;
      return true;
    } else if constexpr (std::is_same<std::optional<int>, T>::value) {
      value_ = std::stoi(*args_it);
      return true;
    } else {
      assert(false);
      return false;
    }
  }

public:
  arg(std::string desc, std::string name)
    requires(P::value)
      : desc_(desc), name_(name), value_() {
    register_arg();
  }
  arg(std::string desc, std::string name, T init = T(),
      std::string value_desc = "")
    requires(!P::value && !std::is_same<bool, T>::value)
      : desc_(desc), name_(name), value_(init), value_desc_(value_desc) {
    register_arg();
  }

  arg(std::string desc, std::string name, bool init = false,
      std::string value_desc = "")
    requires(!P::value && std::is_same<bool, T>::value)
      : desc_(desc), name_(name), value_(init), value_desc_(value_desc) {
    register_arg();
  }

  operator T() const { return value_; }

  const T &operator*() { return value_; }

  const std::string &GetName() const { return name_; }

  const std::string &GetValueDesc() const { return value_desc_; }
  const std::string &GetDesc() const { return desc_; }

  virtual void PrintUsage() const override {
    std::cerr << " ";
    if constexpr (!P::value) {
      std::cerr << "[";
      if constexpr (std::is_same<bool, T>::value) {
        std::cerr << GetName();
      } else {
        std::cerr << GetName() << " " << GetValueDesc();
      }
      std::cerr << "]";
    } else {
      std::cerr << GetName();
    }
  }
  virtual void PrintDesc() const override {
    if constexpr (!P::value) {
      std::cerr << "\t";
      if constexpr (std::is_same<bool, T>::value) {
        std::cerr << GetName();
      } else {
        std::cerr << GetName() << " " << GetValueDesc();
      }
      std::cerr << "\t\t\t" << GetDesc() << "\n";
    } else {
      std::cerr << "\t" << GetName() << "\t\t\t" << GetDesc() << "\n";
    }
  }
};

} // namespace torch_mlir_onnx
