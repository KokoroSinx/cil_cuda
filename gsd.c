// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the General Simulation Data (GSD) project, released under the BSD 2-Clause License.

#ifdef _WIN32

#define GSD_USE_MMAP 0
#include <io.h>
#include <sys/stat.h>

#else // linux / mac

#define _XOPEN_SOURCE 500
#include <unistd.h>
#include <sys/mman.h>
#define GSD_USE_MMAP 1

#endif

#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>

#include "gsd.h"

/*! \file gsd.c
    \brief Implements the GSD C API
*/

// define windows wrapper functions
#ifdef _WIN32
#define lseek _lseeki64
#define write _write
#define read _read
#define open _open
#define ftruncate _chsize

int S_IRUSR = _S_IREAD;
int S_IWUSR = _S_IWRITE;
int S_IRGRP = _S_IREAD;
int S_IWGRP = _S_IWRITE;

size_t pread(int fd, void *buf, size_t count, int64_t offset)
    {
    int64_t oldpos = _telli64(fd);
    _lseeki64(fd, offset, SEEK_SET);
    size_t result = _read(fd, buf, count);  // Note: does not support >4GB reads
    _lseeki64(fd, oldpos, SEEK_SET);
    return result;
    }

size_t pwrite(int fd, void *buf, size_t count, int64_t offset)
    {
    int64_t oldpos = _telli64(fd);
    _lseeki64(fd, offset, SEEK_SET);
    size_t result = _write(fd, buf, count);  // Note: does not support >4GB writes
    _lseeki64(fd, oldpos, SEEK_SET);
    return result;
    }

#endif

/*! \internal
    \brief Utility function to expand the memory space for the index block
    \param handle handle to the open gsd file
*/
static int __gsd_expand_index(struct gsd_handle *handle)
    {
    // multiply the index size each time it grows
    // this allows the index to grow rapidly to accommodate new frames
    const int multiplication_factor = 2;

    // save the old size and update the new size
    size_t old_size = handle->header.index_allocated_entries;
    handle->header.index_allocated_entries = old_size * multiplication_factor;

    if (handle->open_flags == GSD_OPEN_READWRITE)
        {
        // allocate the new larger index block
        handle->index = (struct gsd_index_entry *)realloc(handle->index, sizeof(struct gsd_index_entry) * old_size * multiplication_factor);
        if (handle->index == NULL)
            return -1;

        // zero the new memory
        memset(handle->index + old_size, 0, sizeof(struct gsd_index_entry) * (old_size * multiplication_factor - old_size));

        // now, put the new larger index at the end of the file
        handle->header.index_location = lseek(handle->fd, 0, SEEK_END);
        size_t bytes_written = write(handle->fd, handle->index, sizeof(struct gsd_index_entry) * handle->header.index_allocated_entries);
        if (bytes_written != sizeof(struct gsd_index_entry) * handle->header.index_allocated_entries)
            return -1;

        // set the new file size
        handle->file_size = handle->header.index_location + bytes_written;
        }
    else if (handle->open_flags == GSD_OPEN_APPEND)
        {
        // in append mode, we don't have the whole index stored in memory. Instead, we need to copy it in chunks
        // from the file's old position to the new position
        const size_t buf_size = 1024*16;
        char buf[1024*16];

        int64_t new_index_location = lseek(handle->fd, 0, SEEK_END);
        int64_t old_index_location = handle->header.index_location;
        size_t total_bytes_written = 0;
        size_t old_index_bytes = old_size * sizeof(struct gsd_index_entry);
        while (total_bytes_written < old_index_bytes)
            {
            size_t bytes_to_copy = buf_size;
            if (old_index_bytes - total_bytes_written < buf_size)
                bytes_to_copy = old_index_bytes - total_bytes_written;

            size_t bytes_read = pread(handle->fd, buf, bytes_to_copy, old_index_location + total_bytes_written);
            if (bytes_read != bytes_to_copy)
                return -1;

            size_t bytes_written = pwrite(handle->fd, buf, bytes_to_copy, new_index_location + total_bytes_written);
            if (bytes_written != bytes_to_copy)
                return -1;
            total_bytes_written += bytes_written;
            }

        // fill the new index space with 0s
        memset(buf, 0, buf_size);
        size_t new_index_bytes = old_size * sizeof(struct gsd_index_entry) * multiplication_factor;
        while (total_bytes_written < new_index_bytes)
            {
            size_t bytes_to_copy = buf_size;
            if (new_index_bytes - total_bytes_written < buf_size)
                bytes_to_copy = new_index_bytes - total_bytes_written;

            size_t bytes_written = pwrite(handle->fd, buf, bytes_to_copy, new_index_location + total_bytes_written);
            if (bytes_written != bytes_to_copy)
                return -1;
            total_bytes_written += bytes_written;
            }

        // update to the new index location in the header
        handle->header.index_location = new_index_location;
        handle->file_size = handle->header.index_location + total_bytes_written;
        }

    // write the new header out
    lseek(handle->fd, 0, SEEK_SET);
    size_t bytes_written = write(handle->fd, &(handle->header), sizeof(struct gsd_header));
    if (bytes_written != sizeof(struct gsd_header))
        return -1;

    return 0;
    }

/*! \internal
    \brief utility function to search the namelist and return the id assigned to the name
    \param handle handle to the open gsd file
    \param name string name
    \param append Set to true to allow appending new names into the index, false to disallow

    \return the id assigned to the name, or UINT16_MAX if not found and append is false
*/
uint16_t __gsd_get_id(struct gsd_handle *handle, const char *name, uint8_t append)
    {
    // search for the name in the namelist
    size_t i;
    for (i = 0; i < handle->namelist_num_entries; i++)
        {
        if (0 == strncmp(name, handle->namelist[i].name, sizeof(handle->namelist[i].name)))
            return i;
        }

    // append the name if allowed
    if (append &&
        (handle->open_flags == GSD_OPEN_READWRITE || handle->open_flags == GSD_OPEN_APPEND) &&
        handle->namelist_num_entries < handle->header.namelist_allocated_entries)
        {
        strncpy(handle->namelist[handle->namelist_num_entries].name, name, sizeof(struct gsd_namelist_entry)-1);
        handle->namelist[handle->namelist_num_entries].name[sizeof(struct gsd_namelist_entry)-1] = 0;

        // update the namelist on disk
        lseek(handle->fd,
              handle->header.namelist_location + sizeof(struct gsd_namelist_entry)*handle->namelist_num_entries,
              SEEK_SET);
        size_t bytes_written = write(handle->fd, &(handle->namelist[handle->namelist_num_entries]), sizeof(struct gsd_namelist_entry));
        if (bytes_written != sizeof(struct gsd_namelist_entry))
            return UINT16_MAX;

        handle->namelist_num_entries++;
        return handle->namelist_num_entries-1;
        }
    else
        {
        // otherwise, return not found
        return UINT16_MAX;
        }
    }

/*! \param fd file descriptor to initialize

    Truncate the file and write a new gsd header.
*/
int __gsd_initialize_file(int fd, const char *application, const char *schema, uint32_t schema_version)
    {
    // check if the file was created
    if (fd == -1)
        return -1;

    int retval = ftruncate(fd, 0);
    lseek(fd, 0, SEEK_SET);
    if (retval != 0)
        return retval;

    // populate header fields
    struct gsd_header header;
    memset(&header, 0, sizeof(header));

    header.magic = 0x65DF65DF65DF65DF;
    header.gsd_version = gsd_make_version(1,0);
    strncpy(header.application, application, sizeof(header.application)-1);
    header.application[sizeof(header.application)-1] = 0;
    strncpy(header.schema, schema, sizeof(header.schema)-1);
    header.schema[sizeof(header.schema)-1] = 0;
    header.schema_version = schema_version;
    header.index_location = sizeof(header);
    header.index_allocated_entries = 128;
    header.namelist_location = header.index_location + sizeof(struct gsd_index_entry)*header.index_allocated_entries;
    header.namelist_allocated_entries = 128;
    memset(header.reserved, 0, sizeof(header.reserved));

    // write the header out
    size_t bytes_written = write(fd, &header, sizeof(header));
    if (bytes_written != sizeof(header))
        return -1;

    // allocate and zero default index memory
    struct gsd_index_entry index[128];
    memset(index, 0, sizeof(index));

    // write the empty index out
    bytes_written = write(fd, index, sizeof(index));
    if (bytes_written != sizeof(index))
        return -1;

    // allocate and zero the namelist memory
    struct gsd_namelist_entry namelist[128];
    memset(namelist, 0, sizeof(namelist));

    // write the namelist out
    bytes_written = write(fd, namelist, sizeof(namelist));
    if (bytes_written != sizeof(namelist))
        return -1;

    return 0;
    }

/*! \param handle Handle to read the header

    \pre handle->fd is an open file.
    \pre handle->open_flags is set.

    Read in the file index.
*/
int __gsd_read_header(struct gsd_handle* handle)
    {
    // check if the file was created
    if (handle->fd == -1)
        return -1;

    // read the header
    lseek(handle->fd, 0, SEEK_SET);
    size_t bytes_read = read(handle->fd, &handle->header, sizeof(struct gsd_header));
    if (bytes_read != sizeof(struct gsd_header))
        {
        if (errno != 0)
            return -1;
        else
            return -2;
        }

    // validate the header
    if (handle->header.magic != 0x65DF65DF65DF65DF)
        return -2;

    if (handle->header.gsd_version < gsd_make_version(1,0) && handle->header.gsd_version != gsd_make_version(0,3))
        return -3;
    if (handle->header.gsd_version >= gsd_make_version(2,0))
        return -3;

    // determine the file size
    handle->file_size = lseek(handle->fd, 0, SEEK_END);

    // map the file in read only mode
    #if GSD_USE_MMAP
    if (handle->open_flags == GSD_OPEN_READONLY)
        {
        handle->mapped_data = mmap(NULL, handle->file_size, PROT_READ, MAP_SHARED, handle->fd, 0);

        if (handle->mapped_data == MAP_FAILED)
            return -1;

        handle->index = (struct gsd_index_entry *) (((char *)handle->mapped_data) + handle->header.index_location);
        handle->namelist = (struct gsd_namelist_entry *) (((char *)handle->mapped_data) + handle->header.namelist_location);
        }
    else if (handle->open_flags == GSD_OPEN_READWRITE)
    #endif
        {
        // read the indices into our own memory
        handle->mapped_data = NULL;

        // validate that the index block exists inside the file
        if (handle->header.index_location + sizeof(struct gsd_index_entry) * handle->header.index_allocated_entries > handle->file_size)
            return -4;

        // read the index block
        handle->index = (struct gsd_index_entry *)malloc(sizeof(struct gsd_index_entry) * handle->header.index_allocated_entries);
        if (handle->index == NULL)
            return -5;

        lseek(handle->fd, handle->header.index_location, SEEK_SET);
        bytes_read = read(handle->fd, handle->index, sizeof(struct gsd_index_entry) * handle->header.index_allocated_entries);
        if (bytes_read != sizeof(struct gsd_index_entry) * handle->header.index_allocated_entries)
            return -1;
        }
    #if GSD_USE_MMAP
    else if (handle->open_flags == GSD_OPEN_APPEND)
        {
        // in append mode, we want to avoid reading the entire index in memory, but we also don't want to bother
        // keeping the mapping up to date. Map the index for now to determine index_num_entries, but then
        // unmap it and use different logic to manage a cache of only unwritten index entries
        handle->mapped_data = mmap(NULL, handle->file_size, PROT_READ, MAP_SHARED, handle->fd, 0);

        if (handle->mapped_data == MAP_FAILED)
            return -1;

        handle->index = (struct gsd_index_entry *) (((char *)handle->mapped_data) + handle->header.index_location);
        }
    #endif

    if (handle->open_flags == GSD_OPEN_READWRITE || handle->open_flags == GSD_OPEN_APPEND || !GSD_USE_MMAP)
        {
        // validate that the namelist block exists inside the file
        if (handle->header.namelist_location + sizeof(struct gsd_namelist_entry) * handle->header.namelist_allocated_entries > handle->file_size)
            return -4;

        // read the namelist block
        handle->namelist = (struct gsd_namelist_entry *)malloc(sizeof(struct gsd_namelist_entry) * handle->header.namelist_allocated_entries);
        if (handle->namelist == NULL)
            return -5;

        lseek(handle->fd, handle->header.namelist_location, SEEK_SET);
        bytes_read = read(handle->fd, handle->namelist, sizeof(struct gsd_namelist_entry) * handle->header.namelist_allocated_entries);
        if (bytes_read != sizeof(struct gsd_namelist_entry) * handle->header.namelist_allocated_entries)
            return -1;
        }

    if (handle->index[0].location == 0)
        {
        handle->index_num_entries = 0;
        }
    else
        {
        // determine the number of index entries (marked by location = 0)
        // binary search for the first index entry with location 0
        size_t L = 0;
        size_t R = handle->header.index_allocated_entries;

        // progressively narrow the search window by halves
        do
            {
            size_t m = (L+R)/2;

            if (handle->index[m].location != 0)
                L = m;
            else
                R = m;
            } while ((R-L) > 1);

        // this finds R = the first index entry with location = 0
        handle->index_num_entries = R;
        }

    // determine the number of namelist entries (marked by location = 0)
    // base case: the namelist is full
    handle->namelist_num_entries = handle->header.namelist_allocated_entries;

    // general case, find the first namelist entry that is the empty string
    size_t i;
    for (i = 0; i < handle->header.namelist_allocated_entries; i++)
        {
        if (handle->namelist[i].name[0] == 0)
            {
            handle->namelist_num_entries = i;
            break;
            }
        }


    // determine the current frame counter
    if (handle->index_num_entries == 0)
        {
        handle->cur_frame = 0;
        }
    else
        {
        handle->cur_frame = handle->index[handle->index_num_entries-1].frame + 1;
        }

    // at this point, all valid index entries have been written to disk
    handle->index_written_entries = handle->index_num_entries;

    if (handle->open_flags == GSD_OPEN_APPEND)
        {
        #if GSD_USE_MMAP
        // in append mode, we need to tear down the temporary mapping and allocate a temporary buffer
        // to hold indices for a single frame
        int retval = munmap(handle->mapped_data, handle->file_size);
        if (retval != 0)
            return -1;
        #else
        free(handle->index);
        #endif

        handle->append_index_size = 1;
        handle->index = (struct gsd_index_entry *)malloc(sizeof(struct gsd_index_entry) * handle->append_index_size);
        if (handle->index == NULL)
            return -5;

        handle->mapped_data = NULL;
        }

    return 0;
    }

/*! \param major major version
    \param minor minor version

    \return a packed version number aaaa.bbbb suitable for storing in a gsd file version entry.
*/
uint32_t gsd_make_version(unsigned int major, unsigned int minor)
    {
    return major << 16 | minor;
    }

/*! \param fname File name
    \param application Generating application name (truncated to 63 chars)
    \param schema Schema name for data to be written in this GSD file (truncated to 63 chars)
    \param schema_version Version of the scheme data to be written (make with gsd_make_version())

    \post Create an empty gsd file in a file of the given name. Overwrite any existing file at that location.

    The generated gsd file is not opened. Call gsd_open() to open it for writing.

    \return 0 on success, -1 on a file IO failure - see errno for details
*/
int gsd_create(const char *fname, const char *application, const char *schema, uint32_t schema_version)
    {
    int extra_flags = 0;
    #ifdef WIN32
    extra_flags = _O_BINARY;
    #endif

    // create the file
    int fd = open(fname, O_RDWR | O_CREAT | O_TRUNC | extra_flags,  S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    int retval = __gsd_initialize_file(fd, application, schema, schema_version);
    close(fd);
    return retval;
    }

/*! \param handle Handle to open
    \param fname File name
    \param application Generating application name (truncated to 63 chars)
    \param schema Schema name for data to be written in this GSD file (truncated to 63 chars)
    \param schema_version Version of the scheme data to be written (make with gsd_make_version())
    \param flags Either GSD_OPEN_READWRITE, or GSD_OPEN_APPEND
    \param exclusive_create Set to non-zero to force exclusive creation of the file

    \post Create an empty gsd file in a file of the given name. Overwrite any existing file at that location.

    Open the generated gsd file in *handle*.

    \return 0 on success. Negative value on failure:
        * -1: IO error (check errno)
        * -2: Not a GSD file
        * -3: Invalid GSD file version
        * -4: Corrupt file
        * -5: Unable to allocate memory
        * -6: Invalid argument
*/
int gsd_create_and_open(struct gsd_handle* handle,
                        const char *fname,
                        const char *application,
                        const char *schema,
                        uint32_t schema_version,
                        const enum gsd_open_flag flags,
                        int exclusive_create)
    {
    int extra_flags = 0;
    #ifdef WIN32
    extra_flags = _O_BINARY;
    #endif

    // set the open flags in the handle
    if (flags == GSD_OPEN_READWRITE)
        {
        handle->open_flags = GSD_OPEN_READWRITE;
        }
    else if (flags == GSD_OPEN_READONLY)
        {
        return -6;
        }
    else if (flags == GSD_OPEN_APPEND)
        {
        handle->open_flags = GSD_OPEN_APPEND;
        }

    // set the exclusive create bit
    if (exclusive_create)
        extra_flags |= O_EXCL;

    // create the file
    handle->fd = open(fname, O_RDWR | O_CREAT | O_TRUNC | extra_flags,  S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    int retval = __gsd_initialize_file(handle->fd, application, schema, schema_version);
    if (retval != 0)
        return retval;

    return __gsd_read_header(handle);
    }

/*! \param handle Handle to open
    \param fname File name to open
    \param flags Either GSD_OPEN_READWRITE, GSD_OPEN_READONLY, or GSD_OPEN_APPEND

    \pre The file name \a fname is a GSD file.

    \post Open a GSD file and populates the handle for use by API calls.

    \return 0 on success. Negative value on failure:
        * -1: IO error (check errno)
        * -2: Not a GSD file
        * -3: Invalid GSD file version
        * -4: Corrupt file
        * -5: Unable to allocate memory
*/
int gsd_open(struct gsd_handle* handle, const char *fname, const enum gsd_open_flag flags)
    {
    // allocate the handle
    memset(handle, 0, sizeof(struct gsd_handle));
    handle->index = NULL;
    handle->namelist = NULL;
    handle->cur_frame = 0;

    int extra_flags = 0;
    #ifdef WIN32
    extra_flags = _O_BINARY;
    #endif

    // open the file
    if (flags == GSD_OPEN_READWRITE)
        {
        handle->fd = open(fname, O_RDWR | extra_flags);
        handle->open_flags = GSD_OPEN_READWRITE;
        }
    else if (flags == GSD_OPEN_READONLY)
        {
        handle->fd = open(fname, O_RDONLY | extra_flags);
        handle->open_flags = GSD_OPEN_READONLY;
        }
    else if (flags == GSD_OPEN_APPEND)
        {
        handle->fd = open(fname, O_RDWR | extra_flags);
        handle->open_flags = GSD_OPEN_APPEND;
        }

    return __gsd_read_header(handle);
    }

/*! \param handle Handle to an open GSD file

    Truncate the gsd file, then write a new header. Truncating a file removes all frames and data chunks. The
    application, schema, and schema version are not modified. Truncating may be useful when writing restart files
    to reduce the metadata load on Lustre file servers.

    \return 0 on success. Negative value on failure:
        * -1: IO error (check errno)
        * -2: Invalid input
        * -3: Invalid GSD file version
        * -4: Corrupt file
        * -5: Unable to allocate memory
*/
int gsd_truncate(struct gsd_handle* handle)
    {
    if (handle == NULL)
        return -2;
    if (handle->open_flags == GSD_OPEN_READONLY)
        return -2;

    // deallocate indices
    if (handle->namelist != NULL)
        {
        free(handle->namelist);
        handle->namelist = NULL;
        }

    if (handle->index != NULL)
        {
        free(handle->index);
        handle->index = NULL;
        }

    // keep a copy of the old header
    struct gsd_header old_header = handle->header;
    int retval = __gsd_initialize_file(handle->fd,
                                       old_header.application,
                                       old_header.schema,
                                       old_header.schema_version);

    if (retval != 0)
        return retval;

    return __gsd_read_header(handle);
    }

/*! \param handle Handle to an open GSD file

    \pre \a handle was opened by gsd_open().
    \pre gsd_end_frame() has been called since the last call to gsd_write_chunk().

    \post The file is closed.
    \post \a handle is freed and can no longer be used.

    \warning Do not write chunks to the file with gsd_write_chunk() and then immediately close the file with gsd_close().
    This will result in data loss. Data chunks written by gsd_write_chunk() are not updated in the index until
    gsd_end_frame() is called. This is by design to prevent partial frames in files.

    \return 0 on success, -1 on a file IO failure - see errno for details, and -2 on invalid input
*/
int gsd_close(struct gsd_handle* handle)
    {
    if (handle == NULL)
        return -2;

    // save the fd so we can use it after freeing the handle
    int fd = handle->fd;

    // zero and free memory allocated in the handle
    #if GSD_USE_MMAP
    if (handle->mapped_data != NULL)
        {
        munmap(handle->mapped_data, handle->file_size);
        handle->index = NULL;
        handle->namelist = NULL;

        memset(handle, 0, sizeof(struct gsd_handle));

        // close the file
        int retval = close(fd);
        if (retval != 0)
            return -1;
        }
    else
    #endif
        {
        if (handle->namelist != NULL)
            {
            free(handle->namelist);
            handle->namelist = NULL;
            }

        if (handle->index != NULL)
            {
            free(handle->index);
            handle->index = NULL;

            memset(handle, 0, sizeof(struct gsd_handle));

            // close the file
            int retval = close(fd);
            if (retval != 0)
                return -1;
            }
        }

    return 0;
    }

/*! \param handle Handle to an open GSD file

    \pre \a handle was opened by gsd_open().
    \pre gsd_write_chunk() has been called at least once since the last call to gsd_end_frame().

    \post The current frame counter is increased by 1 and cached indexes are written to disk.

    \return 0 on success, -1 on a file IO failure - see errno for details, and -2 on invalid input
*/
int gsd_end_frame(struct gsd_handle* handle)
    {
    if (handle == NULL)
        return -2;
    if (handle->open_flags == GSD_OPEN_READONLY)
        return -2;

    // all data chunks have already been written to the file and the index updated in memory. To end a frame, all we
    // need to do is increment the frame counter
    handle->cur_frame++;

    // and write unwritten index entries to the file (if there are any to write)
    uint64_t entries_to_write = handle->index_num_entries - handle->index_written_entries;
    if (entries_to_write > 0)
        {
        // write just those unwritten entries to the end of the index block
        int64_t write_pos = handle->header.index_location + sizeof(struct gsd_index_entry)*handle->index_written_entries;

        // in append mode, the start of the write is at the start of the index in memory
        // in readwrite mode, the entire index is in memory, so start at index_written_entries
        struct gsd_index_entry* data = handle->index;
        if (handle->open_flags != GSD_OPEN_APPEND)
            data += handle->index_written_entries;

        size_t bytes_written = pwrite(handle->fd,
                                     data,
                                     sizeof(struct gsd_index_entry)*entries_to_write,
                                     write_pos);

        if (bytes_written != sizeof(struct gsd_index_entry) * entries_to_write)
            return -1;

        handle->index_written_entries += entries_to_write;
        }

    return 0;
    }

/*! \param handle Handle to an open GSD file
    \param name Name of the data chunk (truncated to 63 chars)
    \param type type ID that identifies the type of data in \a data
    \param N Number of rows in the data
    \param M Number of columns in the data
    \param flags set to 0, non-zero values reserved for future use
    \param data Data buffer

    \pre \a handle was opened by gsd_open().
    \pre \a name is a unique name for data chunks in the given frame.
    \pre data is allocated and contains at least `N * M * gsd_sizeof_type(type)` bytes.

    \post The given data chunk is written to the end of the file and its location is updated in the in-memory index.

    \return 0 on success, -1 on a file IO failure - see errno for details, and -2 on invalid input
*/
int gsd_write_chunk(struct gsd_handle* handle,
                    const char *name,
                    enum gsd_type type,
                    uint64_t N,
                    uint32_t M,
                    uint8_t flags,
                    const void *data)
    {
    // validate input
    if (data == NULL)
        return -2;
    if (N == 0 || M == 0)
        return -2;
    if (handle->open_flags == GSD_OPEN_READONLY)
        return -2;

    // populate fields in the index_entry data
    struct gsd_index_entry index_entry;
    memset(&index_entry, 0, sizeof(index_entry));
    index_entry.frame = handle->cur_frame;
    index_entry.id = __gsd_get_id(handle, name, 1);
    index_entry.type = (uint8_t)type;
    index_entry.N = N;
    index_entry.M = M;
    size_t size = N * M * gsd_sizeof_type(type);

    // find the location at the end of the file for the chunk
    index_entry.location = handle->file_size;

    // write the data
    size_t bytes_written = pwrite(handle->fd, data, size, index_entry.location);
    if (bytes_written != size)
        return -1;

    // update the file_size in the handle
    handle->file_size += bytes_written;

    // update the index entry in the index
    // need to expand the index if it is already full
    if (handle->index_num_entries >= handle->header.index_allocated_entries)
        {
        int retval = __gsd_expand_index(handle);
        if (retval != 0)
            return -1;
        }

    // once we get here, there is a free slot to add this entry to the index
    size_t slot = handle->index_num_entries;

    // in append mode, only unwritten entries are stored in memory
    if (handle->open_flags == GSD_OPEN_APPEND)
        {
        slot -= handle->index_written_entries;
        if (slot >= handle->append_index_size)
            {
            handle->append_index_size *= 2;
            handle->index = (struct gsd_index_entry *)realloc(handle->index, handle->append_index_size*sizeof(struct gsd_index_entry));
            if (handle->index == NULL)
                return -1;
            }
        }
    handle->index[slot] = index_entry;
    handle->index_num_entries++;

    return 0;
    }

/*! \param handle Handle to an open GSD file

    \pre \a handle was opened by gsd_open().

    \return The number of frames in the file, or 0 on error
*/
uint64_t gsd_get_nframes(struct gsd_handle* handle)
    {
    if (handle == NULL)
        return 0;
    return handle->cur_frame;
    }

/*! \param handle Handle to an open GSD file
    \param frame Frame to look for chunk
    \param name Name of the chunk to find

    \pre \a handle was opened by gsd_open() in read or readwrite mode.

    \return A pointer to the found chunk, or NULL if not found
*/
const struct gsd_index_entry* gsd_find_chunk(struct gsd_handle* handle, uint64_t frame, const char *name)
    {
    if (handle == NULL)
        return NULL;
    if (name == NULL)
        return NULL;
    if (frame >= gsd_get_nframes(handle))
        return NULL;
    if (handle->open_flags == GSD_OPEN_APPEND)
        return NULL;

    // find the id for the given name
    uint16_t match_id = __gsd_get_id(handle, name, 0);
    if (match_id == UINT16_MAX)
        return NULL;

    // binary search for the first index entry at the requested frame
    size_t L = 0;
    size_t R = handle->index_num_entries;

    // progressively narrow the search window by halves
    do
        {
        size_t m = (L+R)/2;

        if (frame < handle->index[m].frame)
            R = m;
        else
            L = m;
        } while ((R-L) > 1);

    // this finds L = the rightmost index with the desired frame
    int64_t cur_index;

    // search all index entries with the matching frame
    for (cur_index = L; (cur_index >= 0) && (handle->index[cur_index].frame == frame); cur_index--)
        {
        // if the frame matches, check the id
        if (match_id == handle->index[cur_index].id)
            {
            return &(handle->index[cur_index]);
            }
        }

    // if we got here, we didn't find the specified chunk
    return NULL;
    }

/*! \param handle Handle to an open GSD file
    \param data Data buffer to read into
    \param chunk Chunk to read

    \pre \a handle was opened by gsd_open() in read or readwrite mode.
    \pre \a chunk was found by gsd_find_chunk().
    \pre \a data points to an allocated buffer with at least `N * M * gsd_sizeof_type(type)` bytes.

    \return 0 on success, -1 on a file IO failure - see errno for details, and -2 on invalid input
*/
int gsd_read_chunk(struct gsd_handle* handle, void* data, const struct gsd_index_entry* chunk)
    {
    if (handle == NULL)
        return -2;
    if (data == NULL)
        return -2;
    if (chunk == NULL)
        return -2;
    if (handle->open_flags == GSD_OPEN_APPEND)
        return -2;

    size_t size = chunk->N * chunk->M * gsd_sizeof_type(chunk->type);
    if (size == 0)
        return -3;
    if (chunk->location == 0)
        return -3;

    // validate that we don't read past the end of the file
    if ((chunk->location + size) > handle->file_size)
        {
        return -3;
        }

    size_t bytes_read = pread(handle->fd, data, size, chunk->location);
    if (bytes_read != size)
        {
        return -1;
        }

    return 0;
    }

/*! \param type Type ID to query

    \return Size of the given type, or 0 for an unknown type ID.
*/
size_t gsd_sizeof_type(enum gsd_type type)
    {
    if (type == GSD_TYPE_UINT8)
        return 1;
    else if (type == GSD_TYPE_UINT16)
        return 2;
    else if (type == GSD_TYPE_UINT32)
        return 4;
    else if (type == GSD_TYPE_UINT64)
        return 8;
    else if (type == GSD_TYPE_INT8)
        return 1;
    else if (type == GSD_TYPE_INT16)
        return 2;
    else if (type == GSD_TYPE_INT32)
        return 4;
    else if (type == GSD_TYPE_INT64)
        return 8;
    else if (type == GSD_TYPE_FLOAT)
        return 4;
    else if (type == GSD_TYPE_DOUBLE)
        return 8;
    else
        return 0;
    }

// undefine windows wrapper macros
#ifdef _WIN32
#undef lseek
#undef write
#undef read
#undef open
#undef ftruncate

#endif
