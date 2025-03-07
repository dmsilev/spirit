#include <engine/Backend.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/StateType.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Dense>

namespace C = Utility::Constants;

namespace Engine
{

namespace Manifoldmath
{

void project_parallel( vectorfield & vf1, const vectorfield & vf2 )
{
    scalar proj = Vectormath::dot( vf1, vf2 );
    Backend::transform(
        SPIRIT_PAR vf2.begin(), vf2.end(), vf1.begin(),
        [proj] SPIRIT_LAMBDA( const Vector3 & v ) -> Vector3 { return proj * v; } );
}

void project_orthogonal( vectorfield & vf1, const vectorfield & vf2 )
{
    const scalar x = Vectormath::dot( vf1, vf2 );
    Backend::transform(
        SPIRIT_PAR vf1.begin(), vf1.end(), vf2.begin(), vf1.begin(),
        [x] SPIRIT_LAMBDA( const Vector3 & v1, const Vector3 & v2 ) -> Vector3 { return v1 - x * v2; } );
}

void invert_parallel( vectorfield & vf1, const vectorfield & vf2 )
{
    const scalar x = Vectormath::dot( vf1, vf2 );
    Backend::transform(
        SPIRIT_PAR vf1.begin(), vf1.end(), vf2.begin(), vf1.begin(),
        [x] SPIRIT_LAMBDA( const Vector3 & v1, const Vector3 & v2 ) -> Vector3 { return v1 - 2 * x * v2; } );
}

void invert_orthogonal( vectorfield & vf1, const vectorfield & vf2 )
{
    vectorfield vf3 = vf1;
    project_orthogonal( vf3, vf2 );
    Backend::transform(
        SPIRIT_PAR vf1.begin(), vf1.end(), vf3.begin(), vf1.begin(),
        [] SPIRIT_LAMBDA( const Vector3 & v1, const Vector3 & v3 ) -> Vector3 { return v1 - 2 * v3; } );
}

void project_tangential( vectorfield & vf1, const vectorfield & vf2 )
{
    Backend::transform(
        SPIRIT_PAR vf1.begin(), vf1.end(), vf2.begin(), vf1.begin(),
        [] SPIRIT_LAMBDA( const Vector3 & v1, const Vector3 & v2 ) -> Vector3 { return v1 - v1.dot( v2 ) * v2; } );
}

scalar max_tangential_norm( const vectorfield & vector_field, const vectorfield & normal_field )
{
    return sqrt( Backend::transform_reduce(
        SPIRIT_PAR vector_field.begin(), vector_field.end(), normal_field.begin(), scalar( 0 ),
        [] SPIRIT_LAMBDA( const scalar lhs, const scalar rhs ) { return ( lhs < rhs ) ? rhs : lhs; },
        [] SPIRIT_LAMBDA( const Vector3 & v, const Vector3 & n ) { return ( v - v.dot( n ) * n ).squaredNorm(); } ) );
}

scalar dist_geodesic( const vectorfield & v1, const vectorfield & v2 )
{
    return sqrt( Backend::transform_reduce(
        SPIRIT_PAR v1.begin(), v1.end(), v2.begin(), scalar( 0 ), Backend::plus<scalar>{},
        [] SPIRIT_LAMBDA( const Vector3 & v1, const Vector3 & v2 ) -> scalar
        {
            const scalar phi = Vectormath::angle( v1, v2 );
            return phi * phi;
        } ) );
}

/*
    Helper function for a more accurate tangent
*/
void Geodesic_Tangent(
    vectorfield & tangent, const vectorfield & image_1, const vectorfield & image_2, const vectorfield & image_mid )
{
    const auto * image_minus = image_1.data();
    const auto * image_plus  = image_2.data();
    const auto * image_zero  = image_mid.data();
    auto * tang              = tangent.data();

    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), image_1.size(),
        [image_minus, image_plus, image_zero, tang] SPIRIT_LAMBDA( const int idx ) -> void
        {
            const Vector3 ex     = { 1, 0, 0 };
            const Vector3 ey     = { 0, 1, 0 };
            const scalar epsilon = 1e-15;

            Vector3 axis = image_plus[idx].cross( image_minus[idx] );

            // If the spins are anti-parallel, we choose an arbitrary axis
            if( std::abs( image_minus[idx].dot( image_plus[idx] ) + 1 ) < epsilon ) // Check if anti-parallel
            {
                if( std::abs( image_zero[idx].dot( ex ) - 1 ) > epsilon ) // Check if parallel to ex
                    axis = ex;
                else
                    axis = ey;
            }
            tang[idx] = image_zero[idx].cross( axis );
        } );
    Manifoldmath::normalize( tangent );
};

/*
Calculates the 'tangent' vectors, i.e.in crudest approximation the difference between an image and the neighbouring
*/
template<typename StateType>
void Tangents(
    const std::vector<std::shared_ptr<StateType>> & configurations, const std::vector<scalar> & energies,
    std::vector<vectorfield> & tangents )
{
    const auto noi = configurations.size();
    const auto nos = configurations[0]->spin.size();

    if( noi < 2 )
        return;

    // first image
    {
        const auto & image      = configurations[0]->spin;
        const auto & image_plus = configurations[1]->spin;
        Geodesic_Tangent(
            tangents[0], image, image_plus,
            image ); // Use the accurate tangent at the endpoints, useful for the dimer method
    }

    // Images Inbetween
    for( unsigned int idx_img = 1; idx_img < noi - 1; ++idx_img )
    {
        const auto & image       = configurations[idx_img]->spin;
        const auto & image_plus  = configurations[idx_img + 1]->spin;
        const auto & image_minus = configurations[idx_img - 1]->spin;

        // Energies
        scalar E_mid = 0, E_plus = 0, E_minus = 0;
        E_mid   = energies[idx_img];
        E_plus  = energies[idx_img + 1];
        E_minus = energies[idx_img - 1];

        // Vectors to neighbouring images
        vectorfield t_plus( nos ), t_minus( nos );

        Vectormath::set_c_a( 1, image_plus, t_plus );
        Vectormath::add_c_a( -1, image, t_plus );

        Vectormath::set_c_a( 1, image, t_minus );
        Vectormath::add_c_a( -1, image_minus, t_minus );

        // Near maximum or minimum
        if( ( E_plus < E_mid && E_mid > E_minus ) || ( E_plus > E_mid && E_mid < E_minus ) )
        {
            // Get a smooth transition between forward and backward tangent
            scalar E_max = std::max( std::abs( E_plus - E_mid ), std::abs( E_minus - E_mid ) );
            scalar E_min = std::min( std::abs( E_plus - E_mid ), std::abs( E_minus - E_mid ) );

            if( E_plus > E_minus )
            {
                Vectormath::set_c_a( E_max, t_plus, tangents[idx_img] );
                Vectormath::add_c_a( E_min, t_minus, tangents[idx_img] );
            }
            else
            {
                Vectormath::set_c_a( E_min, t_plus, tangents[idx_img] );
                Vectormath::add_c_a( E_max, t_minus, tangents[idx_img] );
            }
        }
        // Rising slope
        else if( E_plus > E_mid && E_mid > E_minus )
        {
            Vectormath::set_c_a( 1, t_plus, tangents[idx_img] );
        }
        // Falling slope
        else if( E_plus < E_mid && E_mid < E_minus )
        {
            Vectormath::set_c_a( 1, t_minus, tangents[idx_img] );
            // tangents = t_minus;
            for( unsigned int i = 0; i < nos; ++i )
            {
                tangents[idx_img][i] = t_minus[i];
            }
        }
        // No slope(constant energy)
        else
        {
            Vectormath::set_c_a( 1, t_plus, tangents[idx_img] );
            Vectormath::add_c_a( 1, t_minus, tangents[idx_img] );
        }

        // Project tangents into tangent planes of spin vectors to make them actual tangents
        project_tangential( tangents[idx_img], image );
        // Normalise in 3N - dimensional space
        Manifoldmath::normalize( tangents[idx_img] );
    }

    // Last Image
    {
        const auto & image       = configurations[noi - 1]->spin;
        const auto & image_minus = configurations[noi - 2]->spin;
        Geodesic_Tangent(
            tangents[noi - 1], image_minus, image,
            image ); // Use the accurate tangent at the endpoints, useful for the dimer method
    }
} // end Tangents

template void Tangents(
    const std::vector<std::shared_ptr<Engine::Spin::StateType>> & configurations, const std::vector<scalar> & energies,
    std::vector<vectorfield> & tangents );

scalar norm( const vectorfield & vf )
{
    scalar x = Vectormath::dot( vf, vf );
    return std::sqrt( x );
}

void normalize( vectorfield & vf )
{
    scalar sc = 1.0 / norm( vf );
    Vectormath::scale( vf, sc );
}

MatrixX tangential_projector( const vectorfield & image )
{
    const auto nos = image.size();
    int size       = 3 * nos;

    // Get projection matrix M=1-S, blockwise S=x*x^T
    MatrixX proj = MatrixX::Identity( size, size );
    for( unsigned int i = 0; i < nos; ++i )
    {
        proj.block<3, 3>( 3 * i, 3 * i ) -= image[i] * image[i].transpose();
    }

    return proj;
}

// This gives an orthogonal matrix of shape (3N, 2N), meaning M^T=M^-1 or M^T*M=1.
// This assumes that the vectors of vf are normalized and that basis is 3N x 2N
// It can be used to transform a vector into or back from the tangent space of a
//      sphere w.r.t. euclidean 3N space.
// It is generated by column-wise normalization of the Jacobi matrix for the
//      transformation from (unit-)spherical coordinates to euclidean.
// It therefore consists of the local basis vectors of the spherical coordinates
//      of a unit sphere, represented in 3N, as the two columns of the matrix.
void tangent_basis_spherical( const vectorfield & vf, MatrixX & basis )
{
    Vector3 tmp, etheta, ephi;
    basis.setZero();
    for( unsigned int i = 0; i < vf.size(); ++i )
    {
        if( vf[i][2] > 1 - 1e-8 )
        {
            tmp                                   = Vector3{ 1, 0, 0 };
            basis.block<3, 1>( 3 * i, 2 * i )     = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            tmp                                   = Vector3{ 0, 1, 0 };
            basis.block<3, 1>( 3 * i, 2 * i + 1 ) = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
        }
        else if( vf[i][2] < -1 + 1e-8 )
        {
            tmp                                   = Vector3{ 1, 0, 0 };
            basis.block<3, 1>( 3 * i, 2 * i )     = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            tmp                                   = Vector3{ 0, -1, 0 };
            basis.block<3, 1>( 3 * i, 2 * i + 1 ) = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
        }
        else
        {
            const scalar rxy   = std::sqrt( 1 - vf[i][2] * vf[i][2] );
            const scalar z_rxy = vf[i][2] / rxy;

            // Note: these are not unit vectors, but derivatives!
            etheta = Vector3{ vf[i][0] * z_rxy, vf[i][1] * z_rxy, -rxy };
            ephi   = Vector3{ -vf[i][1] / rxy, vf[i][0] / rxy, 0 };

            basis.block<3, 1>( 3 * i, 2 * i )     = ( etheta - etheta.dot( vf[i] ) * vf[i] ).normalized();
            basis.block<3, 1>( 3 * i, 2 * i + 1 ) = ( ephi - ephi.dot( vf[i] ) * vf[i] ).normalized();
        }
    }
}

void sparse_tangent_basis_spherical( const vectorfield & vf, SpMatrixX & basis )
{
    std::vector<Eigen::Triplet<scalar>> triplet_list;
    triplet_list.reserve( vf.size() * 3 );

    Vector3 tmp, etheta, ephi, res;
    for( unsigned int i = 0; i < vf.size(); ++i )
    {
        if( vf[i][2] > 1 - 1e-8 )
        {
            tmp = Vector3{ 1, 0, 0 };
            res = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();

            triplet_list.emplace_back( 3 * i, 2 * i, res[0] );
            triplet_list.emplace_back( 3 * i + 1, 2 * i, res[1] );
            triplet_list.emplace_back( 3 * i + 2, 2 * i, res[2] );

            tmp = Vector3{ 0, 1, 0 };
            res = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            triplet_list.emplace_back( 3 * i, 2 * i + 1, res[0] );
            triplet_list.emplace_back( 3 * i + 1, 2 * i + 1, res[1] );
            triplet_list.emplace_back( 3 * i + 2, 2 * i + 1, res[2] );
        }
        else if( vf[i][2] < -1 + 1e-8 )
        {
            tmp = Vector3{ 1, 0, 0 };
            res = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            triplet_list.emplace_back( 3 * i, 2 * i, res[0] );
            triplet_list.emplace_back( 3 * i + 1, 2 * i, res[1] );
            triplet_list.emplace_back( 3 * i + 2, 2 * i, res[2] );

            tmp = Vector3{ 0, -1, 0 };
            res = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            triplet_list.emplace_back( 3 * i, 2 * i + 1, res[0] );
            triplet_list.emplace_back( 3 * i + 1, 2 * i + 1, res[1] );
            triplet_list.emplace_back( 3 * i + 2, 2 * i + 1, res[2] );
        }
        else
        {
            scalar rxy   = std::sqrt( 1 - vf[i][2] * vf[i][2] );
            scalar z_rxy = vf[i][2] / rxy;

            // Note: these are not unit vectors, but derivatives!
            etheta = Vector3{ vf[i][0] * z_rxy, vf[i][1] * z_rxy, -rxy };
            ephi   = Vector3{ -vf[i][1] / rxy, vf[i][0] / rxy, 0 };

            res = ( etheta - etheta.dot( vf[i] ) * vf[i] ).normalized();
            triplet_list.emplace_back( 3 * i, 2 * i, res[0] );
            triplet_list.emplace_back( 3 * i + 1, 2 * i, res[1] );
            triplet_list.emplace_back( 3 * i + 2, 2 * i, res[2] );
            res = ( ephi - ephi.dot( vf[i] ) * vf[i] ).normalized();
            triplet_list.emplace_back( 3 * i, 2 * i + 1, res[0] );
            triplet_list.emplace_back( 3 * i + 1, 2 * i + 1, res[1] );
            triplet_list.emplace_back( 3 * i + 2, 2 * i + 1, res[2] );
        }
    }
    basis.setFromTriplets( triplet_list.begin(), triplet_list.end() );
}

// This calculates the basis via calculation of cross products
// This assumes that the vectors of vf are normalized and that basis is 3N x 2N
void tangent_basis_cross( const vectorfield & vf, MatrixX & basis )
{
    basis.setZero();
    for( unsigned int i = 0; i < vf.size(); ++i )
    {
        if( std::abs( vf[i].z() ) > 1 - 1e-8 )
        {
            basis.block<3, 1>( 3 * i, 2 * i )     = Vector3{ 0, 1, 0 }.cross( vf[i] ).normalized();
            basis.block<3, 1>( 3 * i, 2 * i + 1 ) = vf[i].cross( basis.block<3, 1>( 3 * i, 2 * i ) );
        }
        else
        {
            basis.block<3, 1>( 3 * i, 2 * i )     = Vector3{ 0, 0, 1 }.cross( vf[i] ).normalized();
            basis.block<3, 1>( 3 * i, 2 * i + 1 ) = vf[i].cross( basis.block<3, 1>( 3 * i, 2 * i ) );
        }
    }
}

// This calculates the basis via orthonormalization to a random vector
// This assumes that the vectors of vf are normalized and that basis is 3N x 2N
void tangent_basis_righthanded( const vectorfield & vf, MatrixX & basis )
{
    const auto size = vf.size();
    basis.setZero();

    // vf should be 3N
    // basis should be 3N x 2N

    // e1 and e2 will form a righthanded vectorset with the axis (though not orthonormal!)
    Vector3 e1, e2, v1;
    Vector3 ex{ 1, 0, 0 }, ey{ 0, 1, 0 }, ez{ 0, 0, 1 };

    for( unsigned int i = 0; i < size; ++i )
    {
        const auto & axis = vf[i];

        // Choose orthogonalisation basis for Grahm-Schmidt
        //      We will need two vectors with which the axis always forms the
        //      same orientation (händigkeit des vektor-dreibeins)
        // If axis_z=0 its in the xy-plane
        //      the vectors should be: axis, ez, (axis x ez)
        if( axis[2] == 0 )
        {
            e1 = ez;
            e2 = axis.cross( ez );
        }
        // Else its either above or below the xy-plane.
        //      if its above the xy-plane, it points in z-direction
        //      the vectors should be: axis, ex, -ey
        else if( axis[2] > 0 )
        {
            e1 = ex;
            e2 = -ey;
        }
        //      if its below the xy-plane, it points in -z-direction
        //      the vectors should be: axis, ex, ey
        else if( axis[2] < 0 )
        {
            e1 = ex;
            e2 = ey;
        }

        // First vector: orthogonalize e1 w.r.t. axis
        v1                                = ( e1 - e1.dot( axis ) * axis ).normalized();
        basis.block<3, 1>( 3 * i, 2 * i ) = v1;

        // Second vector: orthogonalize e2 w.r.t. axis and v1
        basis.block<3, 1>( 3 * i, 2 * i + 1 ) = ( e2 - e2.dot( axis ) * axis - e2.dot( v1 ) * v1 ).normalized();
    }
}

// This gives the Jacobian matrix for the transformation from (unit-)spherical
// to euclidean coordinates. It consists of the derivative vectors d/d_theta
// and d/d_phi as the two columns of the matrix.
void spherical_to_cartesian_jacobian( const vectorfield & vf, MatrixX & jacobian )
{
    Vector3 tmp, etheta, ephi;
    jacobian.setZero();
    for( unsigned int i = 0; i < vf.size(); ++i )
    {
        if( vf[i][2] > 1 - 1e-8 )
        {
            tmp                                      = Vector3{ 1, 0, 0 };
            jacobian.block<3, 1>( 3 * i, 2 * i )     = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            tmp                                      = Vector3{ 0, 1, 0 };
            jacobian.block<3, 1>( 3 * i, 2 * i + 1 ) = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
        }
        else if( vf[i][2] < -1 + 1e-8 )
        {
            tmp                                      = Vector3{ 1, 0, 0 };
            jacobian.block<3, 1>( 3 * i, 2 * i )     = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            tmp                                      = Vector3{ 0, -1, 0 };
            jacobian.block<3, 1>( 3 * i, 2 * i + 1 ) = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
        }
        else
        {
            const scalar rxy   = std::sqrt( 1 - vf[i][2] * vf[i][2] );
            const scalar z_rxy = vf[i][2] / rxy;

            // Note: these are not unit vectors, but derivatives!
            etheta = Vector3{ vf[i][0] * z_rxy, vf[i][1] * z_rxy, -rxy };
            ephi   = Vector3{ -vf[i][1], vf[i][0], 0 };

            jacobian.block<3, 1>( 3 * i, 2 * i )     = etheta - etheta.dot( vf[i] ) * vf[i];
            jacobian.block<3, 1>( 3 * i, 2 * i + 1 ) = ephi - ephi.dot( vf[i] ) * vf[i];
        }
    }
}

// The Hessian matrix of the transformation from spherical to euclidean coordinates
void spherical_to_cartesian_hessian( const vectorfield & vf, MatrixX & gamma_x, MatrixX & gamma_y, MatrixX & gamma_z )
{
    const auto nos = vf.size();
    gamma_x.setZero();
    gamma_y.setZero();
    gamma_z.setZero();

    for( unsigned int i = 0; i < nos; ++i )
    {
        scalar z_rxy = vf[i][2] / std::sqrt( 1 + 1e-6 - vf[i][2] * vf[i][2] );

        gamma_x.block<2, 2>( 2 * i, 2 * i ) << -vf[i][0], -vf[i][1] * z_rxy, -vf[i][1] * z_rxy, -vf[i][0];

        gamma_y.block<2, 2>( 2 * i, 2 * i ) << -vf[i][1], vf[i][0] * z_rxy, vf[i][0] * z_rxy, -vf[i][1];

        gamma_z.block<2, 2>( 2 * i, 2 * i ) << -vf[i][2], 0, 0, 0;
    }
}

// The (2Nx2N) Christoffel symbols of the transformation from (unit-)spherical coordinates to euclidean
void spherical_to_cartesian_christoffel_symbols( const vectorfield & vf, MatrixX & gamma_theta, MatrixX & gamma_phi )
{
    const auto nos = vf.size();
    gamma_theta    = MatrixX::Zero( 2 * nos, 2 * nos );
    gamma_phi      = MatrixX::Zero( 2 * nos, 2 * nos );

    for( unsigned int i = 0; i < nos; ++i )
    {
        const scalar theta = acos( vf[i][2] );
        const scalar cot   = abs( theta ) > 1e-4 ? -tan( C::Pi_2 + theta ) : 0;

        gamma_theta( 2 * i + 1, 2 * i + 1 ) = -sin( theta ) * cos( theta );

        gamma_phi( 2 * i + 1, 2 * i ) = cot;
        gamma_phi( 2 * i, 2 * i + 1 ) = cot;
    }
}

void sparse_hessian_bordered_3N(
    const vectorfield & image, const vectorfield & gradient, const SpMatrixX & hessian, SpMatrixX & hessian_out )
{
    // Calculates a 3Nx3N matrix in the bordered Hessian approach and transforms it into the tangent basis,
    // making the result a 2Nx2N matrix. The bordered Hessian's Lagrange multipliers assume a local extremum.

    const auto nos = image.size();
    VectorX lambda( nos );
    for( unsigned int i = 0; i < nos; ++i )
        lambda[i] = image[i].normalized().dot( gradient[i] );

    // Construct hessian_out
    std::vector<Eigen::Triplet<scalar>> tripletList;
    tripletList.reserve( hessian.nonZeros() + 3 * nos );

    // Iterate over non zero entries of hesiian
    for( int k = 0; k < hessian.outerSize(); ++k )
    {
        for( SpMatrixX::InnerIterator it( hessian, k ); it; ++it )
        {
            tripletList.emplace_back( it.row(), it.col(), it.value() );
        }
        const int i = ( k - ( k % 3 ) ) / 3;
        tripletList.emplace_back( k, k, -lambda[i] ); // Correction to the diagonal
    }
    hessian_out.setFromTriplets( tripletList.begin(), tripletList.end() );
}

void hessian_bordered(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & tangent_basis,
    MatrixX & hessian_out )
{
    // Calculates a 3Nx3N matrix in the bordered Hessian approach and transforms it into the tangent basis,
    // making the result a 2Nx2N matrix. The bordered Hessian's Lagrange multipliers assume a local extremum.

    const auto nos = image.size();
    MatrixX tmp_3N = hessian;

    VectorX lambda( nos );
    for( unsigned int i = 0; i < nos; ++i )
        lambda[i] = image[i].dot( gradient[i] );

    for( unsigned int i = 0; i < nos; ++i )
    {
        for( unsigned int j = 0; j < 3; ++j )
        {
            tmp_3N( 3 * i + j, 3 * i + j ) -= lambda( i );
        }
    }

    // Calculate the basis transformation matrix
    tangent_basis = MatrixX::Zero( 3 * nos, 2 * nos );
    tangent_basis_spherical( image, tangent_basis );

    // Result is a 2Nx2N matrix
    hessian_out = tangent_basis.transpose() * tmp_3N * tangent_basis;
}

void hessian_projected(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & tangent_basis,
    MatrixX & hessian_out )
{
    // Calculates a 3Nx3N matrix in the projector approach and transforms it into the tangent basis,
    // making the result a 2Nx2N matrix

    const auto nos = image.size();
    hessian_out.setZero();

    // Calculate projector matrix
    const auto P = tangential_projector( image );

    // Calculate tangential projection of Hessian
    hessian_out = P * hessian * P;

    // Calculate correction terms
    for( unsigned int i = 0; i < nos; ++i )
    {
        hessian_out.block<3, 3>( 3 * i, 3 * i )
            -= P.block<3, 3>( 3 * i, 3 * i ) * ( image[i].dot( gradient[i] ) )
               + ( P.block<3, 3>( 3 * i, 3 * i ) * gradient[i] ) * image[i].transpose();
    }

    // Calculate the basis transformation matrix
    tangent_basis = MatrixX::Zero( 3 * nos, 2 * nos );
    tangent_basis_spherical( image, tangent_basis );

    // Result is a 2Nx2N matrix
    hessian_out = tangent_basis.transpose() * hessian_out * tangent_basis;
}

void hessian_weingarten(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & tangent_basis,
    MatrixX & hessian_out )
{
    // Calculates a 3Nx3N matrix in the Weingarten map approach and transforms it into the tangent basis,
    // making the result a 2Nx2N matrix

    const std::size_t nos = image.size();
    hessian_out.setZero();

    // Calculate projector matrix
    auto P = tangential_projector( image );

    // Calculate tangential projection of Hessian
    hessian_out = P * hessian;

    // Add the Weingarten map
    for( unsigned int i = 0; i < nos; ++i )
    {
        MatrixX proj = MatrixX::Identity( 3, 3 );
        hessian_out.block<3, 3>( 3 * i, 3 * i ) -= MatrixX::Identity( 3, 3 ) * image[i].dot( gradient[i] );
    }

    // Calculate the basis transformation matrix
    tangent_basis = MatrixX::Zero( 3 * nos, 2 * nos );
    tangent_basis_spherical( image, tangent_basis );

    // Result is a 2Nx2N matrix
    hessian_out = tangent_basis.transpose() * hessian_out * tangent_basis;
}

void hessian_spherical(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_out )
{
    // Calculates a 2Nx2N hessian matrix containing second order spherical derivatives

    const auto nos = image.size();

    MatrixX jacobian   = MatrixX::Zero( 3 * nos, 2 * nos );
    MatrixX sph_hess_x = MatrixX::Zero( 2 * nos, 2 * nos );
    MatrixX sph_hess_y = MatrixX::Zero( 2 * nos, 2 * nos );
    MatrixX sph_hess_z = MatrixX::Zero( 2 * nos, 2 * nos );

    // Calculate coordinate transformation jacobian
    Engine::Manifoldmath::spherical_to_cartesian_jacobian( image, jacobian );

    // Calculate coordinate transformation Hessian
    Engine::Manifoldmath::spherical_to_cartesian_hessian( image, sph_hess_x, sph_hess_y, sph_hess_z );

    // Calculate transformed Hessian
    hessian_out = jacobian.transpose() * hessian * jacobian;
    for( unsigned int i = 0; i < nos; ++i )
    {
        hessian_out.block<2, 2>( 2 * i, 2 * i ) += gradient[i][0] * sph_hess_x.block<2, 2>( 2 * i, 2 * i )
                                                   + gradient[i][1] * sph_hess_y.block<2, 2>( 2 * i, 2 * i )
                                                   + gradient[i][2] * sph_hess_z.block<2, 2>( 2 * i, 2 * i );
    }
}

void hessian_covariant(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_out )
{
    // Calculates a 2Nx2N covariant hessian matrix containing second order spherical derivatives
    // and correction terms (containing Christoffel symbols)

    const auto nos = image.size();

    // Calculate coordinate transformation jacobian
    MatrixX jacobian( 3 * nos, 2 * nos );
    Engine::Manifoldmath::spherical_to_cartesian_jacobian( image, jacobian );

    // Calculate the gradient in spherical coordinates
    Eigen::Ref<const VectorX> grad = Eigen::Map<const VectorX>( gradient[0].data(), 3 * nos );
    VectorX gradient_spherical     = jacobian.transpose() * grad;

    // Calculate the Hessian in spherical coordinates
    hessian_spherical( image, gradient, hessian, hessian_out );

    // Calculate the Christoffel symbols for spherical coordinates
    MatrixX christoffel_theta = MatrixX::Zero( 2 * nos, 2 * nos );
    MatrixX christoffel_phi   = MatrixX::Zero( 2 * nos, 2 * nos );
    Engine::Manifoldmath::spherical_to_cartesian_christoffel_symbols( image, christoffel_theta, christoffel_phi );

    // Calculate the covariant Hessian
    for( unsigned int i = 0; i < nos; ++i )
    {
        hessian_out.block<2, 2>( 2 * i, 2 * i )
            -= gradient_spherical[2 * i] * christoffel_theta.block<2, 2>( 2 * i, 2 * i )
               + gradient_spherical[2 * i + 1] * christoffel_phi.block<2, 2>( 2 * i, 2 * i );
    }
}

} // namespace Manifoldmath
} // namespace Engine
