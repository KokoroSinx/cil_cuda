#ifndef INPUT_HPP
#define INPUT_HPP

#include <exception>
#include "Common.hpp"
#include "JsonParser.hpp"
#include "SimIO.hpp"
#include "Units.hpp"
#include "Colony.hpp"
#include "Container.hpp"
#include "MD.hpp"
#include "Random.hpp"

/**
 * Reading in from a configuration json file
 *
 */

struct OutputParams{
  string dirName;
  string prjName;
  bool   energy_record;  // Whether the program calculate and output energy or not.
  bool   div_app_record; // Whether the program output the divided/dead cells or not.
};

using json_parser::json;
extern OutputParams paramsOut;
extern CellColony   colony;

/*!
  Read species parameters
*/
void readSpecs(const json &jiop);

void readInteractions(const json &jiop);

/*!
  Read MD parameters
*/
void readMD(const json &jiop);

/*!
  Read Runtime parameters
 */
void readRunTime(const json &jiop);

/*!
  Read Initial configuration
*/
void readInit(const json &jiop);

void readOutput(const json &jiop);

#endif
