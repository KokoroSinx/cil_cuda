#ifndef JSON_PARSER_HPP
#define JSON_PARSER_HPP

#include <fstream>
#include <string>
#include <exception>
#include <vector>
#include "json.hpp"

namespace json_parser{
  using json = nlohmann::json;

  inline bool exists(const json &root, const std::string &name){
    try{
      return (root.find(name) != root.end());
    }catch(std::exception &e){
      throw std::domain_error(name + " : " + e.what());
    }
  }
  
  inline const json& get_child(const json &root, const std::string &name){
    if (exists(root, name)){
      try{
        return root[name];
      }catch(std::exception &e){
        throw std::domain_error(name + " : " + e.what());
      }
    } else {
      throw std::domain_error("Child " + name + " not found.");
    }
  }

  template<typename T>
  inline T get(const json &root, const std::string &name){
    try{
      return root[name].get<T>();
    }catch(std::exception &e){
      throw std::domain_error(name + " : " + e.what());
    }
  }

  template<typename T>
  inline T get(const json &root, const std::string &name, std::initializer_list<T> acceptable_list){
    try{
      T item = root[name].get<T>();
      bool acceptable = false;
      for(auto &dmy : acceptable_list) if(item == dmy) acceptable = true;
      if(!acceptable){
        std::cerr << "Got : " << item << std::endl;
        std::cerr << "Expected one of : ";
        for(auto &dmy : acceptable_list) std::cerr << dmy << ", ";
        std::cerr << std::endl;

        std::string buffer = "Invalid value";
        throw std::domain_error(buffer);
      }
      return item;
    }catch(std::exception &e){
      throw std::domain_error(name + " : " + e.what());
    }
  }

  inline std::size_t get_size(const json &root, const std::string &name){
    try{
      return root[name].size();
    }catch(std::exception &e){
      throw std::domain_error(name + " : " + e.what());
    }
  }
  template<typename T>
  inline std::vector<T> get_vector(const json &root, const std::string &name, const unsigned int &size){
    try{
      std::vector<T> vec = root[name].get<std::vector<T>>();

      auto dmy_size = vec.size();
      if(dmy_size != size){
        std::string buffer =  "vector should be of size " + std::to_string(size)
        + ". Got " + std::to_string(dmy_size);
        throw std::domain_error(buffer);
      }
      return vec;
    }catch(std::exception &e){
      throw std::domain_error(name + " : " + e.what());
    }
  }

  template<typename T>
  inline void load_data(const json &root, const std::string &name,
    T* A,
    const unsigned int &size
  ){
    try{
      auto vec = get_vector<T>(root, name, size);
      for(auto i = 0u; i < size; i++) A[i] = vec[i];
    }catch(std::exception &e){
      throw std::domain_error(name + " : " + e.what());
    }
  }

  template<typename T>
  inline std::vector<std::vector<T>> get_matrix(const json &root, const std::string &name,
    const unsigned int &size1,
    const unsigned int &size2){
      try{
        std::vector<std::vector<T>> mat = root[name].get<std::vector<std::vector<T>>>();
        if(mat.size() != size1){
          std::string buffer = "matrix should have " + std::to_string(size1)
          + " rows. Got " + std::to_string(mat.size());
          throw std::domain_error(buffer);
        }
        for(auto i = 0u; i < mat.size(); i++){
          if(mat[i].size() != size2){
            std::string buffer = "matrix should have " + std::to_string(size2)
            + " columns. Got " + std::to_string(mat.size()) + " for row " + std::to_string(i);
            throw std::domain_error(buffer);
          }
        }
        return mat;
      }catch(std::exception &e){
        throw std::domain_error(name + " : " + e.what());
      }
    }
    template<typename T>
    inline void load_data(const json &root, const std::string &name,
      T* A,
      const unsigned int &size1,
      const unsigned int &size2
    ){
      try{
        auto mat = get_matrix<T>(root, name, size1, size2);
        for(auto i = 0u; i < size1; i++) for(auto j = 0u; j < size2; j++) A[i*size2 + j] = mat[i][j];
      }catch(std::exception &e){
        throw std::domain_error(name + " : " + e.what());
      }
    }

    inline void parse_file(const char* fname, json &root){
      std::ifstream input_file(fname);
      input_file >> root;
      input_file.close();
      std::cerr << "# Using JsonCons" << std::endl;
    }
    inline void parse(const std::string &raw, json &root){
      root = json::parse(raw);
      std::cerr << "# Using JsonCons" << std::endl;
    }

    inline void dump(const char* fname, const json &root){
      std::ofstream output_file(fname);
      output_file << root.dump(4) << std::endl;
      output_file.close();
    }
  }

  #endif
