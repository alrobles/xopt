#include <Rcpp.h>

#include <xopt/solvers/nls_solver.hpp>
#include <xopt/solvers/trust_region_newton.hpp>

#include <limits>
#include <vector>

namespace {

template <typename T>
T get_or_default(const Rcpp::List& control, const char* key, const T& default_value) {
    if (control.containsElementNamed(key)) {
        return Rcpp::as<T>(control[key]);
    }
    return default_value;
}

} // namespace

//' Trust-region Newton solver via C++
//'
//' @param par Numeric vector of initial parameters.
//' @param fn Objective function returning a scalar.
//' @param gr Optional gradient function returning a numeric vector.
//' @param hvp Optional Hessian-vector product callback taking `(x, v)`.
//' @param control Optional control list (`gtol`, `xtol`, `ftol`, `maxiter`,
//'   `cg_maxiter`, `delta_init`, `delta_max`, `eta`, `boundary_tol`).
//' @return An object of class `"xopt_result"`.
//' @export
// [[Rcpp::export]]
Rcpp::List xopt_tr_newton_cpp(Rcpp::NumericVector par,
                              Rcpp::Function fn,
                              Rcpp::Nullable<Rcpp::Function> gr = R_NilValue,
                              Rcpp::Nullable<Rcpp::Function> hvp = R_NilValue,
                              Rcpp::List control = R_NilValue) {
    std::vector<double> x0(par.begin(), par.end());
    if (x0.empty()) {
        Rcpp::stop("par must be a non-empty numeric vector");
    }

    xopt::second_order::ObjectiveFunction objective = [fn](const std::vector<double>& x) {
        Rcpp::NumericVector xr(x.begin(), x.end());
        return Rcpp::as<double>(fn(xr));
    };

    xopt::second_order::GradientFunction gradient = nullptr;
    if (gr.isNotNull()) {
        Rcpp::Function gr_fn(gr);
        gradient = [gr_fn](const std::vector<double>& x, std::vector<double>& g) {
            Rcpp::NumericVector xr(x.begin(), x.end());
            Rcpp::NumericVector gr_out = Rcpp::as<Rcpp::NumericVector>(gr_fn(xr));
            if (gr_out.size() != static_cast<R_xlen_t>(x.size())) {
                Rcpp::stop("gr must return a numeric vector with length(par) elements");
            }
            g.assign(gr_out.begin(), gr_out.end());
        };
    }

    xopt::second_order::HvpFunction hvp_fn = nullptr;
    if (hvp.isNotNull()) {
        Rcpp::Function hvp_cb(hvp);
        hvp_fn = [hvp_cb](const std::vector<double>& x,
                          const std::vector<double>& v,
                          std::vector<double>& hv) {
            Rcpp::NumericVector xr(x.begin(), x.end());
            Rcpp::NumericVector vr(v.begin(), v.end());
            Rcpp::NumericVector out = Rcpp::as<Rcpp::NumericVector>(hvp_cb(xr, vr));
            if (out.size() != static_cast<R_xlen_t>(x.size())) {
                Rcpp::stop("hvp must return a numeric vector with length(par) elements");
            }
            hv.assign(out.begin(), out.end());
        };
    }

    xopt::solvers::TRNewtonControl ctrl;
    ctrl.gtol = get_or_default<double>(control, "gtol",
                get_or_default<double>(control, "grtol", ctrl.gtol));
    ctrl.xtol = get_or_default<double>(control, "xtol", ctrl.xtol);
    ctrl.ftol = get_or_default<double>(control, "ftol", ctrl.ftol);
    ctrl.maxiter = get_or_default<int>(control, "maxiter", ctrl.maxiter);
    ctrl.cg_maxiter = get_or_default<int>(control, "cg_maxiter", ctrl.cg_maxiter);
    ctrl.delta_init = get_or_default<double>(control, "delta_init",
                      get_or_default<double>(control, "stepmax", ctrl.delta_init));
    ctrl.delta_max = get_or_default<double>(control, "delta_max", ctrl.delta_max);
    ctrl.eta = get_or_default<double>(control, "eta", ctrl.eta);
    ctrl.boundary_tol = get_or_default<double>(control, "boundary_tol", ctrl.boundary_tol);

    xopt::solvers::TRNewtonResult result;
    try {
        result = xopt::solvers::trust_region_newton(x0, objective, gradient, hvp_fn, ctrl);
    } catch (const std::exception& e) {
        Rcpp::stop("xopt_tr_newton_cpp failed: %s", e.what());
    }

    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("par") = Rcpp::wrap(result.par),
        Rcpp::Named("value") = result.value,
        Rcpp::Named("gradient") = Rcpp::wrap(result.gradient),
        Rcpp::Named("convergence") = result.convergence,
        Rcpp::Named("message") = result.message,
        Rcpp::Named("iterations") = result.iterations
    );
    out.attr("class") = "xopt_result";
    return out;
}

//' Nonlinear least squares (Levenberg-Marquardt) via C++
//'
//' @param par Numeric vector of initial parameters.
//' @param residual_fn Residual callback returning a numeric vector.
//' @param jacobian_fn Optional Jacobian callback returning an `m x n` matrix.
//' @param control Optional control list (`ftol`, `xtol`, `gtol`, `maxiter`,
//'   `lambda_init`, `lambda_up`, `lambda_down`, `trace`).
//' @return A list with NLS results (`par`, `value`, `residuals`, `gradient`,
//'   `jacobian`, `vcov`, `iterations`, `convergence`, `message`).
//' @export
// [[Rcpp::export]]
Rcpp::List xopt_nls_cpp(Rcpp::NumericVector par,
                        Rcpp::Function residual_fn,
                        Rcpp::Nullable<Rcpp::Function> jacobian_fn = R_NilValue,
                        Rcpp::List control = R_NilValue) {
    std::vector<double> x0(par.begin(), par.end());
    const int n = static_cast<int>(x0.size());
    if (n == 0) {
        Rcpp::stop("par must be a non-empty numeric vector");
    }

    Rcpp::NumericVector r0 = Rcpp::as<Rcpp::NumericVector>(residual_fn(par));
    const int m = static_cast<int>(r0.size());
    if (m < n) {
        Rcpp::stop("Number of residuals must be >= number of parameters");
    }

    xopt::solvers::ResidualFunction residual = [residual_fn](const std::vector<double>& x,
                                                              std::vector<double>& r) {
        Rcpp::NumericVector xr(x.begin(), x.end());
        Rcpp::NumericVector rout = Rcpp::as<Rcpp::NumericVector>(residual_fn(xr));
        r.assign(rout.begin(), rout.end());
    };

    xopt::solvers::JacobianFunction jacobian = nullptr;
    if (jacobian_fn.isNotNull()) {
        Rcpp::Function jac_fn(jacobian_fn);
        jacobian = [jac_fn, m, n](const std::vector<double>& x, std::vector<double>& J) {
            Rcpp::NumericVector xr(x.begin(), x.end());
            Rcpp::NumericMatrix Jm = Rcpp::as<Rcpp::NumericMatrix>(jac_fn(xr));
            if (Jm.nrow() != m || Jm.ncol() != n) {
                Rcpp::stop("jacobian_fn must return an m x n matrix");
            }
            const std::size_t m_sz = static_cast<std::size_t>(m);
            const std::size_t n_sz = static_cast<std::size_t>(n);
            if (n_sz > 0 && m_sz > (std::numeric_limits<std::size_t>::max() / n_sz)) {
                Rcpp::stop("jacobian size overflow");
            }
            J.resize(m_sz * n_sz);
            for (int i = 0; i < m; ++i) {
                const std::size_t row_offset = static_cast<std::size_t>(i) * n_sz;
                for (int j = 0; j < n; ++j) {
                    J[row_offset + static_cast<std::size_t>(j)] = Jm(i, j);
                }
            }
        };
    }

    xopt::solvers::LMControl ctrl;
    ctrl.ftol = get_or_default<double>(control, "ftol", ctrl.ftol);
    ctrl.xtol = get_or_default<double>(control, "xtol", ctrl.xtol);
    ctrl.gtol = get_or_default<double>(control, "gtol", ctrl.gtol);
    ctrl.maxiter = get_or_default<int>(control, "maxiter", ctrl.maxiter);
    ctrl.lambda_init = get_or_default<double>(control, "lambda_init", ctrl.lambda_init);
    ctrl.lambda_up = get_or_default<double>(control, "lambda_up", ctrl.lambda_up);
    ctrl.lambda_down = get_or_default<double>(control, "lambda_down", ctrl.lambda_down);
    ctrl.trace = get_or_default<bool>(control, "trace", ctrl.trace);

    xopt::solvers::NLSResult result;
    try {
        result = xopt::solvers::levenberg_marquardt(x0, residual, jacobian, ctrl);
    } catch (const std::exception& e) {
        Rcpp::stop("xopt_nls_cpp failed: %s", e.what());
    }

    Rcpp::NumericMatrix Jout(result.nresiduals, result.nparams);
    for (int i = 0; i < result.nresiduals; ++i) {
        const std::size_t row_offset = static_cast<std::size_t>(i) * result.nparams;
        for (int j = 0; j < result.nparams; ++j) {
            Jout(i, j) = result.jacobian[row_offset + static_cast<std::size_t>(j)];
        }
    }

    Rcpp::NumericMatrix Vout(result.nparams, result.nparams);
    for (int i = 0; i < result.nparams; ++i) {
        const std::size_t row_offset = static_cast<std::size_t>(i) * result.nparams;
        for (int j = 0; j < result.nparams; ++j) {
            Vout(i, j) = result.vcov[row_offset + static_cast<std::size_t>(j)];
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("par") = Rcpp::wrap(result.par),
        Rcpp::Named("value") = result.value,
        Rcpp::Named("residuals") = Rcpp::wrap(result.residuals),
        Rcpp::Named("gradient") = Rcpp::wrap(result.gradient),
        Rcpp::Named("jacobian") = Jout,
        Rcpp::Named("vcov") = Vout,
        Rcpp::Named("iterations") = result.iterations,
        Rcpp::Named("convergence") = result.convergence,
        Rcpp::Named("message") = result.message
    );
}
