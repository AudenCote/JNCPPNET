#ifndef LOGGER_INCLUDE
#define LOGGER_INCLUDE

#include <iostream>
#include <windows.h>  

HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

class Logger{

public:

  static void Info(const char* str){
    SetConsoleTextAttribute(hConsole, 2);
    std::cout << '\n' << str << '\n' << std::endl;
    SetConsoleTextAttribute(hConsole, 15);
  }

  static void Warning(const char* str){
    SetConsoleTextAttribute(hConsole, 6);
    std::cout << '\n' << str << '\n' << std::endl;
    SetConsoleTextAttribute(hConsole, 15);
  }

  static void Error(const char* str){
    SetConsoleTextAttribute(hConsole, 4);
    std::cout << '\n' << str << '\n' << std::endl;
    SetConsoleTextAttribute(hConsole, 15);
  }
};

#endif