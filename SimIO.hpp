#ifndef SIMIO_HPP
#define SIMIO_HPP

#include <vector>
#include <stack>
#include <iostream>
#include <cassert>
#include <cstring>
#include <hdf5.h>
#include <hdf5_hl.h>


inline void h5_check_err(const herr_t &err){
  if(err < 0){
    std::cerr << "# HDF5 ERROR!" << std::endl;
    H5Eprint(H5E_DEFAULT, stderr);
    exit(1);
  }
}

namespace SimIO{
  using io_space =  hid_t;
  using io_size  =  hsize_t;

  namespace io_select_op{
    enum io_select_op:int {SET, OR, AND, XOR, NOT};
  }
  
  static H5S_seloper_t get_hdf5_select_op(const io_select_op::io_select_op &op){
    if (op == io_select_op::SET){
      return H5S_SELECT_SET;
    }else if( op == io_select_op::OR){
      return H5S_SELECT_OR;
    }else if(op == io_select_op::AND){
      return H5S_SELECT_AND;
    }else if(op == io_select_op::XOR){
      return H5S_SELECT_AND;
    }else if(op == io_select_op::NOT){
      return H5S_SELECT_NOTB;
    }else{
      assert(false);
    }
    return H5S_SELECT_SET;
  }

  //!_______________________________________________________________________________________
  //!     map types to HDF5 types
  //!     \author lg (04 March 2013)
  //!     http://stackoverflow.com/questions/7412042/hdf5-c-interface-writing-dynamic-2d-arrays
  //!_______________________________________________________________________________________ 
  template<typename T> struct get_hdf5_data_type{
    static hid_t type();
    static hid_t type_out();
  };
  template<> struct get_hdf5_data_type<char>        { hid_t type() { return H5T_NATIVE_CHAR;    };  hid_t type_out() { return H5T_NATIVE_CHAR;   }; }; 
  template<> struct get_hdf5_data_type<int8_t>      { hid_t type() { return H5T_NATIVE_INT8;    };  hid_t type_out() { return H5T_NATIVE_INT8;   }; };
  template<> struct get_hdf5_data_type<uint8_t>     { hid_t type() { return H5T_NATIVE_UINT8;   };  hid_t type_out() { return H5T_NATIVE_UINT8;  }; };
  template<> struct get_hdf5_data_type<int16_t>     { hid_t type() { return H5T_NATIVE_INT16;   };  hid_t type_out() { return H5T_NATIVE_INT16;  }; };
  template<> struct get_hdf5_data_type<uint16_t>    { hid_t type() { return H5T_NATIVE_UINT16;  };  hid_t type_out() { return H5T_NATIVE_UINT16; }; };
  template<> struct get_hdf5_data_type<int32_t>     { hid_t type() { return H5T_NATIVE_INT32;   };  hid_t type_out() { return H5T_NATIVE_INT32;  }; };
  template<> struct get_hdf5_data_type<uint32_t>    { hid_t type() { return H5T_NATIVE_UINT32;  };  hid_t type_out() { return H5T_NATIVE_UINT32; }; };
  template<> struct get_hdf5_data_type<int64_t>     { hid_t type() { return H5T_NATIVE_INT64;   };  hid_t type_out() { return H5T_NATIVE_INT64;  }; };
  template<> struct get_hdf5_data_type<uint64_t>    { hid_t type() { return H5T_NATIVE_UINT64;  };  hid_t type_out() { return H5T_NATIVE_UINT64; }; };
  template<> struct get_hdf5_data_type<float>       { hid_t type() { return H5T_NATIVE_FLOAT;   };  hid_t type_out() { return H5T_NATIVE_FLOAT;  }; };
  template<> struct get_hdf5_data_type<double>      { hid_t type() { return H5T_NATIVE_DOUBLE;  };  hid_t type_out() { return H5T_NATIVE_FLOAT;  }; };
  template<> struct get_hdf5_data_type<long double> { hid_t type() { return H5T_NATIVE_LDOUBLE; };  hid_t type_out() { return H5T_NATIVE_FLOAT;  }; };

  namespace Space{
    /*!
      \brief Create new dataspace and add to dataspace vector
    */
    template<typename T>
    inline io_space create(const int &rank, const T *dim){
      hsize_t *dmy_dim = (hsize_t*) malloc(sizeof(hsize_t)*rank);
      for(int i = 0; i < rank; i++) dmy_dim[i] = static_cast<hsize_t>(dim[i]);
      io_space dmy = H5Screate_simple(rank, dmy_dim, NULL);
      h5_check_err(dmy);
      free(dmy_dim);
      return dmy;
    }
    template<typename T>
    inline io_space create(T nx){
      return create(1, &nx);
    }
    template<typename T>
    inline io_space create(T nx, T ny){
      T dim[2] = {nx, ny};
      return create(2, dim);
    }
    template<typename T>
    inline io_space create(T nx, T ny, T nz){
      T dim[3] = {nx, ny, nz};
      return create(3, dim);
    }

    inline void close(io_space &id){
      herr_t status;
      status = H5Sclose(id);
      h5_check_err(status);
    }

    /*!
      \brief Get total size of dataspace
     */
    inline io_size get_size(const io_space &id){
      return H5Sget_select_npoints(id);
    }

    /*!
      \brief Rest dataspace selection
     */
    inline void reset(const io_space &id){
      herr_t status = H5Sselect_none(id);
      h5_check_err(status);
    }
    
    /*!
      \brief Apply hyperslab selection to dataspace
    */
    inline void select_hyperslab(const io_space &space_id, const io_select_op::io_select_op &op,
				 const io_size *start, const io_size *stride, const io_size *count, const io_size *block){
      herr_t status = H5Sselect_hyperslab(space_id,
					  get_hdf5_select_op(op),
					  start, stride, count, block
					  );
      h5_check_err(status);
    }
  };

  class SimIO{
    protected:
    std::stack<hid_t>  loc;
    hid_t              group_create_plist; //track order in which groups are created
  public:

    SimIO(){
      group_create_plist = H5Pcreate(H5P_GROUP_CREATE);
      herr_t status = H5Pset_link_creation_order(group_create_plist, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
      h5_check_err(status);
    }
    ~SimIO(){
      H5Pclose(group_create_plist);
    }

    /*!
      \brief Create new file , overwrite existing file
     */
    inline void create_file(const char* fname){
      assert(loc.empty());
      loc.push(H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
      h5_check_err(loc.top());
    }

    /*!
      \brief Open existing file
     */
    inline void open_file(const char* fname, const char* mode){
      assert(loc.empty());
      hid_t access_mode;
      if(strcmp(mode, "r") == 0){
	access_mode = H5F_ACC_RDONLY;
      }else if(strcmp(mode, "rw") == 0){
	access_mode = H5F_ACC_RDWR;
      }else{
	std::cerr << "# HDF5 FILE ACCESS ERROR!" << std::endl;
	exit(1);	
      }
      loc.push(H5Fopen(fname, access_mode, H5P_DEFAULT));
      h5_check_err(loc.top());
    }

    /*!
      \brief Create new group, add to stack
     */
    inline void create_group(const char* name){
      assert(!loc.empty());
      loc.push(H5Gcreate(loc.top(), name, H5P_DEFAULT, group_create_plist, H5P_DEFAULT));
      h5_check_err(loc.top());
    }

    /*
      \brief Open existing group, add to stack
     */
    inline void open_group(const char* name){
      assert(!loc.empty());
      loc.push(H5Gopen(loc.top(), name, H5P_DEFAULT));
      h5_check_err(loc.top());
    }
    
    /*!
      \brief Flushes group contents to file
     */
    inline void flush_group(){
      assert(!loc.empty());
      herr_t status = H5Fflush(loc.top(), H5F_SCOPE_LOCAL);
      h5_check_err(status);
    }
    
    /*!
      \brief Close group at top of stack
     */
    inline void close_group(){
      assert(loc.size() > 1);
      herr_t status = H5Gclose(loc.top());
      h5_check_err(status);
      loc.pop();
    }
    /*!
      \brief Close all groups except root "/"
     */
    inline void close_groups(){
      herr_t status;
      while(loc.size() > 1){
	status = H5Gclose(loc.top());
	h5_check_err(status);
	loc.pop();
      }
    }

    /*!
      \brief Close file and all open groups assigned to it
     */
    inline void close_file(){
      this->close_groups();
      assert(loc.size() == 1);
      herr_t status = H5Fclose(loc.top());
      h5_check_err(status);
      loc.pop();
    }

    /*!
      \brief Write int attribute to current group
     */
    inline void write_attr(const char *name, const int &attr){
      assert(!loc.empty());
      herr_t status = H5LTset_attribute_int(loc.top(), "./", name, &attr, 1);
      h5_check_err(status);
    }
    inline void write_attr(const char *name, const int* attr, const int &size){
      assert(!loc.empty());      
      herr_t status = H5LTset_attribute_int(loc.top(), "./", name, attr, size);
      h5_check_err(status);
    }


    /*!
      \brief Write unsigned int attribute to current group
    */
    inline void write_attr(const char *name, const unsigned int &attr){
      assert(!loc.empty());
      herr_t status = H5LTset_attribute_uint(loc.top(), "./", name, &attr, 1);
      h5_check_err(status);
    }
    inline void write_attr(const char *name, const unsigned int *attr, const int &size){
      assert(!loc.empty());
      herr_t status = H5LTset_attribute_uint(loc.top(), "./", name, attr, size);
      h5_check_err(status);
    }
    

    /*!
      \brief Write float attribute to current group
     */
    inline void write_attr(const char *name, const float &attr){
      assert(!loc.empty());      
      herr_t status = H5LTset_attribute_float(loc.top(), "./", name, &attr, 1);
      h5_check_err(status);
    }
    inline void write_attr(const char *name, const float* attr, const int &size){
      assert(!loc.empty());      
      herr_t status = H5LTset_attribute_float(loc.top(), "./", name, attr, size);
      h5_check_err(status);
    }
    
    /*!
      \brief Write double attribute to current group
     */
    inline void write_attr(const char *name, const double &attr){
      assert(!loc.empty());      
      herr_t status = H5LTset_attribute_double(loc.top(), "./", name, &attr, 1);
      h5_check_err(status);
    }
    inline void write_attr(const char *name, const double* attr, const int &size){
      assert(!loc.empty());      
      herr_t status = H5LTset_attribute_double(loc.top(), "./", name, attr, size);
      h5_check_err(status);
    }
    /*!
      \brief Write char attribute to current group
     */
    inline void write_attr(const char *name, const char* attr){
      assert(!loc.empty());      
      herr_t status = H5LTset_attribute_string(loc.top(), "./", name, attr);
      h5_check_err(status);
    }


    /*!
      \brief Look for attribute in current group (return rank size if found, otherwise return 0)
     */
    inline int find_attr(const char* name){
      assert(!loc.empty());      
      int rank = 0;
      if(H5LTfind_attribute(loc.top(), name)) H5LTget_attribute_ndims(loc.top(), "./", name, &rank);
      return rank;
    }

    /*!
      \brief Read attribute for current group
     */
    template<typename T>
    inline void read_attr(const char *name, T* attr){
      assert(!loc.empty());      
      get_hdf5_data_type<T> h5_dtype;      
      herr_t status = H5LTget_attribute(loc.top(), "./", name, h5_dtype.type(), attr);
      h5_check_err(status);
    }
    template<typename First, typename...Rest>
    void read_attr(const char* name, First* first, Rest* ... rest){
      read_attr(name, first);
      read_attr(rest...);
    }

    /*!
      \brief Check for the existence of links
    */
    inline bool link_exists(const char *name){
      htri_t status =  H5Lexists(loc.top(), name, H5P_DEFAULT);
      h5_check_err(status);
      return static_cast<bool>(status);
    }    
    
    /*!
      \brief Write data to file
      \details Access memory using mem_space, acess disk using out_space
      \warning Uses default converters when writing to disk (i.e., all real data saved as float!)
     */
    template<typename T>
    inline void write_data(const char* name, hid_t &mem_space, hid_t &out_space, const T* data){
      assert(!loc.empty());      
      get_hdf5_data_type<T> h5_dtype;
      
      hid_t out_dataset = H5Dcreate(loc.top(), name, h5_dtype.type_out(), out_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      h5_check_err(out_dataset);
      
      herr_t status = H5Dwrite(out_dataset, h5_dtype.type(), mem_space, H5S_ALL, H5P_DEFAULT, data);
      h5_check_err(status);
      
      status = H5Dclose(out_dataset);
      h5_check_err(status);
    }
    /*!
      \brief Write data to file
      \details Access memory and disk using mem_out_space
     */
    template<typename T>
    inline void write_data(const char* name, hid_t &mem_out_space, const T* data){
      write_data<T>(name, mem_out_space, mem_out_space, data);
    }

    /*
      \brief Write data to file
      \details Access memory using mem_space, acess disk using out_space
               dmy_data used to determine output type for more control over saved data
     */
    template<typename T, typename U>
    inline void write_data(const char* name, const hid_t &mem_space, const hid_t &out_space, const T* data, const U* dmy_data){
      assert(!loc.empty());            
      get_hdf5_data_type<T> h5_mem_dtype;
      get_hdf5_data_type<U> h5_out_dtype;

      hid_t out_dataset = H5Dcreate(loc.top(), name, h5_out_dtype.type(), out_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      h5_check_err(out_dataset);

      herr_t status = H5Dwrite(out_dataset, h5_mem_dtype.type(), mem_space, H5S_ALL, H5P_DEFAULT, data);
      h5_check_err(status);

      status = H5Dclose(out_dataset);
      h5_check_err(status);
    }
    /*
      \brief Write data to file
      \details Access memory and disk using mem_out_space
               dmy_data used to determine output type for more control over saved data
     */
    template<typename T, typename U>
    inline void write_data(const char* name, hid_t &mem_out_space, const T* data, const U* dmy_data){
      write_data(name, mem_out_space, mem_out_space, data, dmy_data);
    }

    /*
      \brief Read data from file
      \details Access memory using mem_space, access disk using out_space
     */
    template<typename T>
    inline void read_data(const char* name, hid_t &mem_space, hid_t &out_space, T* data){
      assert(!loc.empty());            
      get_hdf5_data_type<T> h5_dtype;

      hid_t out_dataset = H5Dopen(loc.top(), name, H5P_DEFAULT);
      h5_check_err(out_dataset);

      hid_t status = H5Dread(out_dataset, h5_dtype.type(), mem_space, out_space, H5P_DEFAULT, data);
      h5_check_err(status);

      status = H5Dclose(out_dataset);
      h5_check_err(status);
    }

    /*
      \brief Read data from file
      \details Acess memory using mem_space, acess full dataspace on disk
     */
    template<typename T>
    inline void read_data(const char* name, hid_t &mem_space, T*data){
      assert(!loc.empty());            
      get_hdf5_data_type<T> h5_dtype;

      hid_t out_dataset = H5Dopen(loc.top(), name, H5P_DEFAULT);
      h5_check_err(out_dataset);

      hid_t status = H5Dread(out_dataset, h5_dtype.type(), mem_space, H5S_ALL, H5P_DEFAULT, data);
      h5_check_err(status);

      status = H5Dclose(out_dataset);
      h5_check_err(status);
    }

    /*
      \brief Read data from file
      \details Acess full dataspace on disk, assume same layout in memory
     */
    template<typename T>
    inline void read_data(const char* name, T* data){
      assert(!loc.empty());            
      get_hdf5_data_type<T> h5_dtype;

      hid_t out_dataset = H5Dopen(loc.top(), name, H5P_DEFAULT);
      h5_check_err(out_dataset);

      hid_t status = H5Dread(out_dataset, h5_dtype.type(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
      h5_check_err(status);

      status = H5Dclose(out_dataset);
      h5_check_err(status);
    }
    template<typename First, typename...Rest>
    void read_data(const char* name, First* first, Rest* ... rest){
      read_data(name, first);
      read_data(rest...);
    }

    inline void read_data_ndims(const char* name, std::vector<uint> &dims){
      int                  rank;
      std::vector<hsize_t> ddims;
      H5T_class_t          dtype;
      size_t               bytes;
      hid_t                status;
      if((status = H5LTget_dataset_ndims(loc.top(), name, &rank)) >= 0){
	h5_check_err(status);	
	ddims.resize(rank);
	dims.resize(rank);
	status = H5LTget_dataset_info(loc.top(), name, ddims.data(), &dtype, &bytes);
	h5_check_err(status);
	for(int i = 0; i < rank; i++) dims[i] = ddims[i];
      }else{
	dims.clear();
      }
    }
    
    /*
      \brief Write multidimensional real data field DUOBLE -> FLOAT
     */
    inline void write_ndfield(const char* name, hid_t &mem_space, hid_t &out_space, const void* data){
      assert(!loc.empty());            
      hid_t out_dataset = H5Dcreate(loc.top(), name, H5T_NATIVE_FLOAT, out_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      h5_check_err(out_dataset);

      hid_t status = H5Dwrite(out_dataset, H5T_NATIVE_DOUBLE, mem_space, H5S_ALL, H5P_DEFAULT, data);
      h5_check_err(status);

      status = H5Dclose(out_dataset);
      h5_check_err(status);
    }
    inline void write_ndfield(const char* name, hid_t &mem_out_space, const void* data){
      write_ndfield(name, mem_out_space, mem_out_space, data);
    }
  };

}

#endif
