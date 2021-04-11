#include "misc.h"

std::string pad(int x) {
	std::stringstream s;
	if      (x < 10)   s << "00" << x;
	else if (x < 100)  s << "0"  << x;
	else if (x < 1000) s << ""   << x;
	return s.str();
}

std::string number_format(int x) {
	std::stringstream s;
	std::vector<int> sep; int t; bool flag = false;
	if (x < 0) { flag = true; x = -x; }
	while (x > 1000) {
		t = x % 1000;
		sep.push_back(t);
		x = x / 1000;
	};
	sep.push_back(x);
	for (int i = sep.size() - 1; i > 0; i--) {
		if (i == sep.size() - 1) s << (flag ? "-" : "") << sep[i] << ",";
		else s << pad(sep[i]) << ",";
	}
	if (sep.size() > 1)
		s << pad(sep[0]);
	else
		s << (flag ? "-" : "") << sep[0];
	return s.str();
}

