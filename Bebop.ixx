#define OUT
//#define INTEGRATE_EULER
//#define BEBOP_RELEASE

export module Bebop;

import <cmath>;
import <array>;
import <stack>;
import <vector>;
import <variant>;
import <stdexcept>;
import <algorithm>;

import SlotMap;

export namespace Bebop
{
#pragma region Constants
	constexpr int	NOT_AVAILABLE{ -1 };
	constexpr float PI{ 3.14159f };
	constexpr float	DegToRad{ PI / 180.0f };
	constexpr float RadToDeg{ 180.0f / PI };
#pragma endregion

#pragma region Matrix2x2
	// Row major notation; so (m01) == row 0, column 1
	struct Matrix2x2
	{
		Matrix2x2() = default;
		Matrix2x2(float _rotationRadians);
		Matrix2x2(float m00_, float m01_, float m10_, float m11_);

		void	Set(float m00_, float m01_, float m10_, float m11_);

		float	m00{ 0.0f };	float	m01{ 0.0f };
		float	m10{ 0.0f };	float	m11{ 0.0f };
	};

	Matrix2x2::Matrix2x2(float _rotationRadians)
	{
		const float sin = sinf(_rotationRadians);
		const float cos = cosf(_rotationRadians);

		Set(cos, sin, -sin, cos);
	}

	Matrix2x2::Matrix2x2(float m00_, float m01_, float m10_, float m11_)
		: m00{ m00_ }, m01{ m01_ }, m10{ m10_ }, m11{ m11_ }
	{
	}

	void Matrix2x2::Set(float m00_, float m01_, float m10_, float m11_)
	{
		m00 = m00_;
		m01 = m01_;
		m10 = m10_;
		m11 = m11_;
	}
#pragma endregion

#pragma region VecN
	template<int N>
	struct Vec
	{
		float& operator[](int n)
		{
#ifndef BEBOP_RELEASE
			if (n < 0 || n >= N)
			{
				throw std::out_of_range("Index out of range");
			}
#endif
			return _array[n];
		}

		const float& operator[](int n) const
		{
#ifndef BEBOP_RELEASE
			if (n < 0 || n >= N)
			{
				throw std::out_of_range("Index out of range");
			}
#endif

			return _array[n];
		}

	private:
		std::array<float, N> _array;
	};
#pragma endregion

#pragma region MatrixNxM
	template<int Row, int Col>
	struct Mat
	{
		Mat();
		Mat(std::initializer_list<float> l);

		const float* begin() const { return &_array[0]; }
		const float* end() const { return &_array[0] + (Row * Col); }

		float& At(int row, int col) { return _array[row * Col + col]; }
		const float& At(int row, int col) const { return _array[row * Col + col]; }

		Mat<Col, Row>	Transpose() const;
		int				Rows() const { return Row; }
		int				Cols() const { return Col; }

		Mat<Row, Col>	operator*(float f) const;

	private:
		std::array<float, Row* Col> _array;
	};

	template<int Row, int Col>
	Bebop::Mat<Row, Col>::Mat()
	{
		_array.fill(0.0f);
	}

	template<int Row, int Col>
	Mat<Row, Col> Mat<Row, Col>::operator*(float f) const
	{
		Mat<Row, Col> result;

		for (int i = 0; i < Row * Col; ++i)
		{
			result._array[i] = _array[i] * f;
		}

		return result;
	}

	template<int Row, int Col>
	Mat<Row, Col>::Mat(std::initializer_list<float> l)
	{
		std::size_t index = 0;
		for (const auto& item : l)
		{
			_array[index++] = item;
		}
	}

	template<int Row, int Col>
	Mat<Col, Row> Mat<Row, Col>::Transpose() const
	{
		Mat<Col, Row> transposed;

		for (int i = 0; i < Row; ++i)
		{
			for (int j = 0; j < Col; ++j)
			{
				transposed.At(j, i) = At(i, j);
			}
		}

		return transposed;
	}

	template<int RowA, int ColA, int RowB, int ColB>
	Mat<RowA, ColB> operator*(const Mat<RowA, ColA>& a, const Mat<RowB, ColB>& b)
	{
		static_assert(ColA == RowB, "[Bebop] Trying to multiply matrices of invalid dimensions");

		Mat<RowA, ColB> result;

		for (int i = 0; i < RowA; ++i)
		{
			for (int j = 0; j < ColB; ++j)
			{
				float product{ 0.0f };

				for (int k = 0; k < ColA; ++k)
				{
					product += (a.At(i, k) * b.At(k, j));
				}

				result.At(i, j) = product;
			}
		}

		return result;
	}
#pragma endregion

#pragma region Solve linear system of equations using Gauss-Seidel method
	template<int Row, int Col>
	Mat<Col, 1> SolveGaussSeidel(const Mat<Row, Col>& a, const Mat<Row, 1>& b)
	{
		Mat<Col, 1> x;

		const int maxIterations = Col > 6 ? Col : 6;
		for (int iterations = 0; iterations < maxIterations; iterations++)
		{
			for (int i = 0; i < Col; ++i)
			{
				const float a_ = a.At(i, i);
				if (a_ > 0.0f || a_ < 0.0f)
				{
					float dot = 0.0f;
					for (int j = 0; j < Col; ++j)
					{
						dot += (a.At(i, j) * (x.At(j, 0)));
					}

					const float b_ = b.At(i, 0);
					const float dx = (b.At(i, 0) / a_) - (dot / a_);

					// just in case dx is NaN

					if (dx == dx)
					{
						x.At(i, 0) = x.At(i, 0) + dx;
					}
				}
			}
		}

		return x;
	}
#pragma endregion

#pragma region Vec2
	struct Vec2
	{
		float		x{ 0.0f };
		float		y{ 0.0f };

		Vec2() = default;
		Vec2(float x_, float y_) : x{ x_ }, y{ y_ } {}
		Vec2(int x_, int y_) : x{ static_cast<float>(x_) }, y{ static_cast<float>(y_) } {}

		Vec2		operator-() const;
		Vec2&		operator+=(const Vec2& rhs);
		Vec2&		operator-=(const Vec2& rhs);
		Vec2		operator*(const Matrix2x2& m) const;

		float		Magnitude() const;
		float		SqrMagnitude() const;
		float		Cross(const Vec2& rhs) const;
		float		Dot(const Vec2& rhs) const;

		Vec2		GetPerpendicular() const;
		Vec2		NormalizedSafe() const;

		bool		NormalizeSafe();
	};

	float Vec2::Cross(const Vec2& rhs) const
	{
		return (x * rhs.y) - (y * rhs.x);
	}

	float Vec2::Dot(const Vec2& rhs) const
	{
		return x * rhs.x + y * rhs.y;
	}

	Vec2 Vec2::operator-() const
	{
		return Vec2(-x, -y);
	}

	Vec2 Vec2::operator*(const Matrix2x2& m) const
	{
		return Vec2{ x * m.m00 + y * m.m10, x * m.m01 + y * m.m11 };
	}

	float Dot(const Vec2& a, const Vec2& b)
	{
		return a.x * b.x + a.y * b.y;
	}

	Vec2 operator*(const Vec2& a, float s)
	{
		return Vec2(a.x * s, a.y * s);
	}

	Vec2 operator*(float s, const Vec2 a)
	{
		return Vec2(a.x * s, a.y * s);
	}

	Vec2 operator/(const Vec2& a, float s)
	{
		return Vec2(a.x / s, a.y / s);
	}

	Vec2 operator+(const Vec2& a, const Vec2& b)
	{
		return Vec2(a.x + b.x, a.y + b.y);
	}

	Vec2 operator-(const Vec2& a, const Vec2& b)
	{
		return Vec2(a.x - b.x, a.y - b.y);
	}

	Vec2& Vec2::operator+=(const Vec2& rhs)
	{
		x += rhs.x;
		y += rhs.y;

		return *this;
	}

	Vec2& Vec2::operator-=(const Vec2& rhs)
	{
		x -= rhs.x;
		y -= rhs.y;

		return *this;
	}

	float Vec2::Magnitude() const
	{
		return sqrtf(SqrMagnitude());
	}

	float Vec2::SqrMagnitude() const
	{
		return x * x + y * y;
	}

	Vec2 Vec2::GetPerpendicular() const
	{
		return Vec2(-y, x);
	}

	bool Vec2::NormalizeSafe()
	{
		const float sqrMagnitude = SqrMagnitude();

		if (sqrMagnitude > 0.0f)
		{
			const float magnitude = sqrtf(sqrMagnitude);

			x = x / magnitude;
			y = y / magnitude;

			return true;
		}

		x = 0.0f;
		y = 1.0f;

		return false;
	}

	Vec2 Vec2::NormalizedSafe() const
	{
		Vec2 normal{ x, y };
		normal.NormalizeSafe();
		return normal;
	}

#pragma endregion

#pragma region Particle
	struct World;

	struct Particle
	{
		friend struct World;

		Vec2		position{ 0.0f, 0.0f };
		Vec2		previousPosition{ 0.0f, 0.0f };
		Vec2		velocity{ 0.0f, 0.0f };

		float		_inverseMass{ 1.0f };

		Particle() = default;
		Particle(const Particle& other) = default;
		Particle(Particle&& other) = default;

		Particle& operator=(const Particle& other) = default;
		Particle& operator=(Particle&& other) = default;

		Particle(float x, float y) : position{ x, y }, previousPosition{ x, y } {}
		Particle(float x, float y, float mass) : position{ x, y }, previousPosition{ x, y }, _inverseMass{ mass > 0.0f ? 1.0f / mass : 0.0f } {}
		Particle(const Vec2& p) : position{ p }, previousPosition{ p } {}
		Particle(const Vec2& p, float mass) : position{ p }, previousPosition{ p }, _inverseMass{ mass > 0.0f ? 1.0f / mass : 0.0f } {}

		void inline	AddForce(const Vec2& force);
		void inline	Update(float deltaT, const Vec2& gravity);

	private:
		Vec2		_acceleration{ 0.0f, 0.0f };
		Vec2		_forces{ 0.0f, 0.0f };
	};

	Vec2 GenerateSpringForce(const Vec2& anchor,
		const Vec2& otherPoint,
		float restLength,
		float k)
	{
		const Vec2 spring = otherPoint - anchor;
		const float currentLength = spring.Magnitude();
		const float displacement = currentLength - restLength;

		Vec2 springDirection{ 0.0f, 1.0f };
		if (currentLength > 0.0f)
		{
			springDirection = spring / currentLength;
		}

		const float springMagnitude = -k * displacement;
		return springDirection * springMagnitude;
	}

	Vec2 GenerateParticleDragForce(const Particle& particle, float dragConstant)
	{
		auto velocity = particle.velocity;
		const auto speedSquared = velocity.SqrMagnitude();
		if (velocity.NormalizeSafe())
		{
			return -velocity * speedSquared * dragConstant;
		}

		return Vec2();
	}

	void Particle::AddForce(const Vec2& force)
	{
		_forces += force;
	}

	void Particle::Update(float deltaT, const Vec2& gravity)
	{
		_acceleration = _forces * _inverseMass;

		if (_inverseMass > 0.0f)
		{
			_acceleration += gravity;
		}

		_forces.x = 0.0f;
		_forces.y = 0.0f;

#ifdef INTEGRATE_EULER
		velocity += _acceleration * deltaT;
		position += velocity * deltaT;
#else
		const float deltaSquared = deltaT * deltaT;
		const Vec2 prevPosition = position;

		position.x = (position.x * 2.0f) - previousPosition.x + (_acceleration.x * deltaSquared);
		position.y = (position.y * 2.0f) - previousPosition.y + (_acceleration.y * deltaSquared);

		previousPosition = prevPosition;

		velocity = deltaT > 0.0f ? (position - previousPosition) / deltaT : Vec2(0.0f, 0.0f);
#endif
	}
#pragma endregion

#pragma region Spring mesh
	struct Spring
	{
		Spring() = default;
		Spring(Unalmas::SlotMapKey a_,
			Unalmas::SlotMapKey b_,
			Unalmas::SlotMap<Bebop::Particle>* slotmap_,
			float lengthAtRest_,
			float k_) :
			a{ a_ }, b{ b_ },
			slotmap{ slotmap_ },
			lengthAtRest{ lengthAtRest_ }, k{ k_ }
		{}

		float					lengthAtRest{ 1.0f };
		float					k{ 0.0f };

		Bebop::Particle&		A() const { return (*slotmap)[a]; }
		Bebop::Particle&		B() const { return (*slotmap)[b]; }

	private:
		Unalmas::SlotMapKey					a;
		Unalmas::SlotMapKey					b;
		Unalmas::SlotMap<Bebop::Particle>*	slotmap;
	};

#pragma region Contact
	struct Contact
	{
		struct Rigidbody*	a{ nullptr };
		struct Rigidbody*	b{ nullptr };

		Vec2				normal{ 0.0f, 0.0f };
		Vec2				start{ 0.0f, 0.0f };
		Vec2				end{ 0.0f, 0.0f };

		float				depth{ 0.0f };

		// Contacts are expected to be temporary objects. The bodies they point to
		// are not owned by the contact, and they can be moved around in memory after
		// the contact has been created. Therefore the contact should not be used
		// after it's been processed. To help catch errors related to this, you can
		// Invalidate() it after it's been processed.
		void Invalidate()
		{
			a = nullptr;
			b = nullptr;
		}
	};
#pragma endregion

#pragma region Constraints, interface
	enum class ConstraintType
	{
		None = 0,
		ParticleDistance = 1
	};

	struct ParticleConstraint
	{
		ParticleConstraint() = default;
		ParticleConstraint(ConstraintType type_, const Unalmas::SlotMapKey a_,
			const Unalmas::SlotMapKey b_, float param_) :
			type{ type_ }, a{ a_ }, b{ b_ }, param{ param_ } {}

		ConstraintType					type{ ConstraintType::None };
		Unalmas::SlotMapKey				a;
		Unalmas::SlotMapKey				b;
		float							param{ 0.0f };
	};

	struct Rigidbody;
	struct Constraint
	{
		Constraint() = default;
		Constraint(Unalmas::SlotMap<Rigidbody>* bodies_,
			Unalmas::SlotMapKey a_,
			Unalmas::SlotMapKey b_) :
			bodies{ bodies_ },
			a{ a_ },
			b{ b_ }
		{}

		Unalmas::SlotMap<Rigidbody>* bodies{ nullptr };
		Unalmas::SlotMapKey a;
		Unalmas::SlotMapKey b;

		Mat<1, 1>				cachedLambda;

		float					biasFactor{ 0.0f };

		Mat<6, 1>				GetVelocities() const;
		Mat<6, 6>				GetInverseMassMatrix() const;

		virtual void			PreSolve(float deltaT);
		virtual void			Solve();
		virtual void			PostSolve();
	};

	struct JointConstraint : public Constraint
	{
		JointConstraint() = default;
		JointConstraint(Unalmas::SlotMap<Rigidbody>* bodies_,
			Unalmas::SlotMapKey a_,
			Unalmas::SlotMapKey b_,
			const Vec2& anchorPoint);

		Vec2					anchorA;	// Anchor point in a's local space
		Vec2					anchorB;	// Anchor point in b's local space

		Mat<1, 6>				jacobian;

		virtual void			PreSolve(float deltaT) override;
		virtual void			Solve() override;
		virtual void			PostSolve() override;
	};

	struct NonPenetrationConstraint : public Constraint
	{
		NonPenetrationConstraint() = default;
		NonPenetrationConstraint(Unalmas::SlotMap<Rigidbody>* bodies_,
			Unalmas::SlotMapKey a_,
			Unalmas::SlotMapKey b_,
			const Contact& contact);

		Vec2					contactPointA;		// contact point on A, in A's local space
		Vec2					contactPointB;		// contact point on B, in B's local space
		Vec2					contactNormal;		// normal in the local space of A
		Mat<2, 6>				jacobian;

		virtual void			PreSolve(float deltaT) override;
		virtual void			Solve() override;
		virtual void			PostSolve() override;

	private:
		float					_friction{ 0.0f };
	};

#pragma endregion

#pragma region Rigidbody
	enum class RigidbodyType
	{
		Undefined = 0,
		Circle = 1,
		Box = 2,
		Polygon = 3
	};

	struct Shape
	{
	public:
		virtual RigidbodyType	GetType() const = 0;
	};

	struct Circle : public Shape
	{
		Circle() = default;
		Circle(float radius_) : radius{ radius_ } {}

		float	radius{ 1.0f };

	protected:
		virtual RigidbodyType	GetType() const override { return RigidbodyType::Circle; }
	};

	struct Box : public Shape
	{
		Box() = default;
		Box(float width, float height);

		Vec2& operator[](int index);
		const Vec2& operator[](int index) const;
		void		RefreshWorldSpaceVertices(const Matrix2x2& rotationMatrix, const Vec2& position);

		Vec2		v0ws;			// vertex 0 in world space
		Vec2		v1ws;			// vertex 1 in world space
		Vec2		v2ws;			// vertex 2 in world space
		Vec2		v3ws;			// vertex 3 in world space

		// Return the number of clipped points
		int ClipSegmentToLine(const std::vector<Vec2>& contactsIn,
			OUT std::vector<Vec2>& contactsOut,
			Vec2 c0,
			Vec2 c1) const
		{
			int numOut = 0;

			// The cross product is used here, because a.Cross(b) == a.Dot(bPerp),
			// and that's what we're interested in. However this is only true if
			// bPerp is chosen carefully (there are 2 valid interpretations).
			//
			// Because of the way I handle things - assuming a clockwise winding
			// order, I need to swap c1 and c0 here. Maybe revisit this later.

			std::swap(c1, c0);

			const Vec2 normal = (c1 - c0).NormalizedSafe();

			float dist0 = (contactsIn[0] - c0).Cross(normal);
			float dist1 = (contactsIn[1] - c0).Cross(normal);

			// If the points are behind the plane - no need to clip them
			if (dist0 <= 0.0f) { contactsOut[numOut++] = contactsIn[0]; }
			if (dist1 <= 0.0f) { contactsOut[numOut++] = contactsIn[1]; }

			// If the points are on different sides of the plane:
			if (dist0 * dist1 < 0.0f)
			{
				const float totalDist = dist0 - dist1;
				const float t = dist0 / totalDist;
				const Vec2 contact = contactsIn[0] + (contactsIn[1] - contactsIn[0]) * t;
				contactsOut[numOut++] = contact;
			}

			return numOut;
		}

	protected:
		virtual RigidbodyType	GetType() const override { return RigidbodyType::Box; }
		Vec2					_v0;	// vertex 0 in local space
		Vec2					_v1;	// vertex 1 in local space
		Vec2					_v2;	// vertex 2 in local space
		Vec2					_v3;	// vertex 3 in local space
	};

	void Box::RefreshWorldSpaceVertices(const Matrix2x2& rotationMatrix, const Vec2& position)
	{
		v0ws = (_v0 * rotationMatrix) + position;
		v1ws = (_v1 * rotationMatrix) + position;
		v2ws = (_v2 * rotationMatrix) + position;
		v3ws = (_v3 * rotationMatrix) + position;
	}

	Vec2& Box::operator[](int index)
	{
		switch (index)
		{
		case 0: return v0ws;
		case 1: return v1ws;
		case 2: return v2ws;
		case 3: return v3ws;
		default:
			throw std::invalid_argument("Invalid box vertex index");
		}
	}

	const Vec2& Box::operator[](int index) const
	{
		switch (index)
		{
		case 0: return v0ws;
		case 1: return v1ws;
		case 2: return v2ws;
		case 3: return v3ws;
		default:
			throw std::invalid_argument("Invalid box vertex index");
		}
	}

	Box::Box(float width, float height)
	{
		const float halfW = width / 2.0f;
		const float halfH = height / 2.0f;

		_v0 = Vec2{ -halfW, halfH };
		_v1 = Vec2{ halfW, halfH };
		_v2 = Vec2{ halfW, -halfH };
		_v3 = Vec2{ -halfW, -halfH };
	}

	struct Polygon : public Shape
	{
		std::vector<Vec2> vertices;

	protected:
		virtual RigidbodyType	GetType() const override { return RigidbodyType::Polygon; }
	};

	struct Rigidbody
	{
		friend struct World;
		friend struct Constraint;

		Rigidbody() = default;
		Rigidbody(float radius, float mass);
		Rigidbody(float width, float height, float mass);

		RigidbodyType										GetType() const { return _type; }

		float												previousRotation{ 0.0f };
		Vec2												position{ 0.0f, 0.0f };
		Vec2												previousPosition{ 0.0f, 0.0f };
		Vec2												linearVelocity{ 0.0f, 0.0f };
		Vec2												surfaceFriction{ 0.6f, 0.3f };		// static, kinetic
		float												angularVelocity{ 0.0f };
		float												restitution{ 0.9f };


		bool isStatic{ true };

		std::variant<std::monostate, Circle, Box, Polygon>	shape;

		float												GetRadius() const;
		float												GetMass() const;
		float												GetRotationRadians() const;
		Vec2												GetInverseMassAndMomentOfInertia() const;
		float												GetStaticFriction() const { return surfaceFriction.x; }
		float												GetKineticFriction() const { return surfaceFriction.y; }

		void												AddForce(float x, float y);
		void												AddForce(const Vec2& f);
		void												AddLinearImpulse(float x, float y);
		void												AddLinearImpulse(const Vec2& j);
		void												AddAngularImpulse(float j);
		void												AddImpulseAtPoint(const Vec2& j, const Vec2& r);
		void												AddTorque(float t);

		//void												Update(float deltaT, const Vec2& gravity);
		void												IntegrateForces(float deltaT, const Vec2& gravity);
		void												IntegrateVelocities(float deltaT);

		void												SetPosition(const Vec2& position);
		void												SetRotation(float _rotationRadians);
		void												RefreshWorldSpaceTransform();
		void												TryRefreshRotationMatrix();

		Vec2												WorldToLocalSpace(const Vec2& pointWorldSpace) const;
		Vec2												LocalToWorldSpace(const Vec2& pointInLocalSpace) const;

	private:
		Vec2			_linearAcceleration{ 0.0f, 0.0f };
		Vec2			_forces{ 0.0f, 0.0f };
		Vec2			_linearImpulses{ 0.0f, 0.0f };
		Vec2			_positionAdjustment{ 0.0f, 0.0f };
		float			_angularImpulses{ 0.0f };
		float			_rotationRadians{ 0.0f };

		float			_angularAcceleration{ 0.0f };
		float			_torque{ 0.0f };
		float			_inverseMass{ 0.0f };
		float			_inverseMomentOfInertia{ 0.0f };
		RigidbodyType	_type{ RigidbodyType::Undefined };
		Matrix2x2		_rotationMatrix{ 1.0f, 0.0f, 0.0f, 1.0f };
	};

	Rigidbody::Rigidbody(float radius, float mass) :
		_inverseMass{ mass > 0.0f ? 1.0f / mass : 0.0f },
		isStatic{ mass > 0.0f },
		shape{ Circle(radius) },
		_inverseMomentOfInertia{ mass > 0.0f ? 1.0f / ((mass / 2.0f) * radius * radius) : 0.0f },
		_type{ RigidbodyType::Circle }
	{

	}

	Rigidbody::Rigidbody(float width, float height, float mass) :
		_inverseMass{ mass > 0.0f ? 1.0f / mass : 0.0f },
		isStatic{ mass > 0.0f },
		shape{ Box(width, height) },
		_inverseMomentOfInertia{ mass > 0.0f ? 1.0f / ((mass / 12.0f) * (4.0f * height * height + width * width)) : 0.0f },
		_type{ RigidbodyType::Box }
	{
	}

	void Rigidbody::SetRotation(float rotationRadians)
	{
		_rotationRadians = rotationRadians;
		TryRefreshRotationMatrix();
	}

	float Rigidbody::GetRotationRadians() const
	{
		return _rotationRadians;
	}

	Vec2 Rigidbody::LocalToWorldSpace(const Vec2& pointLocalSpace) const
	{
		const Vec2 rotated = pointLocalSpace * _rotationMatrix;
		return rotated + position;
	}

	Vec2 Rigidbody::WorldToLocalSpace(const Vec2& pointWorldSpace) const
	{
		// Rotation matrix:
		// [cos		sin]
		// [-sin	cos]

		// rotation inverse, in this case - transposed:
		// [cos		-sin]
		// [sin		cos]
		const Vec2 translated = pointWorldSpace - position;

		const float cosW = _rotationMatrix.m00;
		const float sinW = _rotationMatrix.m01;

		const float rotatedX = translated.x * cosW + translated.y * sinW;
		const float rotatedY = -translated.x * sinW + translated.y * cosW;

		return Vec2{ rotatedX, rotatedY };
	}

	// Impulse is instantaneous change in velocity, inversely
	// proportional to the mass of the body.
	// Momentum: P = m * v
	// Impulse: J is the change in momentum: J = dP = m * dV
	// Therefore, the change in (linear) velocity is dV = J / m,
	// or in our case, J * inverseMass.

	void Rigidbody::AddLinearImpulse(float x, float y)
	{
		_linearImpulses += Vec2{ x, y };
	}

	void Rigidbody::AddAngularImpulse(float j)
	{
		_angularImpulses += j;
	}

	void Rigidbody::AddLinearImpulse(const Vec2& j)
	{
		_linearImpulses += j;
	}

	void Rigidbody::AddImpulseAtPoint(const Vec2& j, const Vec2& r)
	{
		_linearImpulses += j;
		_angularImpulses += (r.x * j.y - r.y * j.x);
	}

	void Rigidbody::AddTorque(float t)
	{
		_torque += t;
	}

	void Rigidbody::AddForce(const Vec2& f)
	{
		_forces += f;
	}

	void Rigidbody::AddForce(float x, float y)
	{
		_forces += Vec2(x, y);
	}

	float Rigidbody::GetMass() const
	{
		return _inverseMass <= 0.0f ? 0.0f : 1.0f / _inverseMass;
	}

	Vec2 Rigidbody::GetInverseMassAndMomentOfInertia() const
	{
		return Vec2{ _inverseMass, _inverseMomentOfInertia };
	}

	float Rigidbody::GetRadius() const
	{
		return std::get<Circle>(shape).radius;
	}

	void Rigidbody::SetPosition(const Vec2& position)
	{
		this->position = position;
		this->previousPosition = position;
	}

	void Rigidbody::TryRefreshRotationMatrix()
	{
		if (_rotationRadians != previousRotation)
		{
			_rotationMatrix = Matrix2x2(_rotationRadians);
		}
	}

	void Rigidbody::RefreshWorldSpaceTransform()
	{
		TryRefreshRotationMatrix();

		switch (_type)
		{
		case RigidbodyType::Circle:
			break;
		case RigidbodyType::Box:
			std::get<Box>(shape).RefreshWorldSpaceVertices(_rotationMatrix, position);
			break;

			// TODO polygon
		}
	}

	void Rigidbody::IntegrateForces(float deltaT, const Vec2& gravity)
	{
		if (_inverseMass <= 0.0f)
		{
			return;
		}

		_forces += (gravity / _inverseMass);
		_linearAcceleration = _forces * _inverseMass;
		_angularAcceleration = _torque * _inverseMomentOfInertia;

#ifdef INTEGRATE_EULER
		const Vec2 impulseLinearVelocity = _linearImpulses * _inverseMass;
		const float impulseAngularVelocity = _angularImpulses * _inverseMomentOfInertia;

		linearVelocity += _linearAcceleration * deltaT;
		linearVelocity += impulseLinearVelocity;

		angularVelocity += _angularAcceleration * deltaT;
		angularVelocity += impulseAngularVelocity;

		_linearImpulses = Vec2();
		_angularImpulses = 0.0f;
#endif

		_torque = 0.0f;
		_forces = Vec2();
	}

	void Rigidbody::IntegrateVelocities(float deltaT)
	{
		if (_inverseMass <= 0.0f)
		{
			return;
		}

#ifdef INTEGRATE_EULER
		const Vec2 impulseLinearVelocity = _linearImpulses * _inverseMass;
		const float impulseAngularVelocity = _angularImpulses * _inverseMomentOfInertia;

		linearVelocity += impulseLinearVelocity;
		angularVelocity += impulseAngularVelocity;

		position += (linearVelocity * deltaT + _positionAdjustment);
		_rotationRadians += angularVelocity * deltaT;
#else
		const Vec2 impulseLinearVelocity = _linearImpulses * _inverseMass;
		const float impulseAngularVelocity = _angularImpulses * _inverseMomentOfInertia;

		const float deltaSquared = deltaT * deltaT;
		const Vec2 prevPosition = position;

		position.x = (position.x * 2.0f) - previousPosition.x + (_linearAcceleration.x * deltaSquared) + impulseLinearVelocity.x * deltaT;
		position.y = (position.y * 2.0f) - previousPosition.y + (_linearAcceleration.y * deltaSquared) + impulseLinearVelocity.y * deltaT;

		previousPosition = prevPosition;

		linearVelocity = deltaT > 0.0f ? (position - previousPosition) / deltaT : Vec2(0.0f, 0.0f);

		position += _positionAdjustment;
		previousPosition += _positionAdjustment;

		const float prevRotation = _rotationRadians;
		_rotationRadians = _rotationRadians * 2.0f - previousRotation + _angularAcceleration * deltaSquared + impulseAngularVelocity * deltaT;
		previousRotation = prevRotation;

		angularVelocity = deltaT > 0.0f ? (_rotationRadians - previousRotation) / deltaT : 0.0f;
#endif

		_linearImpulses = Vec2();
		_angularImpulses = 0.0f;
		_positionAdjustment = Vec2();
	}

#pragma endregion

#pragma region Constraints implementation
	void Constraint::PreSolve(float deltaT)	// virtual
	{
	}

	void Constraint::Solve()	// virtual
	{
	}

	void Constraint::PostSolve() // virtual
	{
	}

	JointConstraint::JointConstraint(Unalmas::SlotMap<Rigidbody>* bodies_,
		Unalmas::SlotMapKey a_,
		Unalmas::SlotMapKey b_,
		const Vec2& anchorPoint) :
		Constraint{ bodies_, a_, b_ },
		anchorA{ (*bodies_)[a_].WorldToLocalSpace(anchorPoint) },
		anchorB{ (*bodies_)[b_].WorldToLocalSpace(anchorPoint) }
	{
	}

	NonPenetrationConstraint::NonPenetrationConstraint(Unalmas::SlotMap<Rigidbody>* bodies_,
		Unalmas::SlotMapKey a_,
		Unalmas::SlotMapKey b_,
		const Contact& contact)
		:
		Constraint{ bodies_, a_, b_ },
		contactPointA{ (*bodies_)[a].WorldToLocalSpace(contact.end) },
		contactPointB{ (*bodies_)[b].WorldToLocalSpace(contact.start) },
		contactNormal{ (*bodies_)[a].WorldToLocalSpace(contact.normal) }
	{
	}

	void NonPenetrationConstraint::PreSolve(float deltaT)
	{
		//	C' = JV + b = 0,
		//	where C' = the first time derivative of the position constraint function
		//		  J  = the jacobian
		//		  V  = the velocities vector
		//		  b  = the bias (smoothing) term

		// How to calculate J in the case of non-penetration constraint?
		// [??? ??? ??? ???] * [ va, wa, vb, wb]
		//
		// where: 
		//		  
		//		  
		//		  
		//		  

		// Get collision points and normal in world space
		const auto& aBody = (*bodies)[a];
		const auto& bBody = (*bodies)[b];
		const Vec2 pa = aBody.LocalToWorldSpace(contactPointA);		// pa should be the collision endpoint
		const Vec2 pb = bBody.LocalToWorldSpace(contactPointB);		// pb should the collision start point
		const Vec2 n = aBody.LocalToWorldSpace(contactNormal);

		const Vec2 ra = pa - aBody.position;
		const Vec2 rb = pb - bBody.position;

		jacobian.At(0, 0) = -(n.x);
		jacobian.At(0, 1) = -(n.y);
		jacobian.At(0, 2) = (-ra).Cross(n);

		jacobian.At(0, 3) = n.x;
		jacobian.At(0, 4) = n.y;
		jacobian.At(0, 5) = rb.Cross(n);

		// Calculate relative velocity pre-impulse normal (to calculate elasticity)
		const Vec2 va = aBody.linearVelocity + Vec2(-aBody.angularVelocity * ra.y, aBody.angularVelocity * ra.x);
		const Vec2 vb = bBody.linearVelocity + Vec2(-bBody.angularVelocity * rb.y, bBody.angularVelocity * rb.x);
		const float vRelativeAlongNormal = Dot(va - vb, n);
		const float e = std::min(aBody.restitution, bBody.restitution);

		_friction = std::max(aBody.GetKineticFriction(), bBody.GetKineticFriction());
		if (_friction > 0.0f)
		{
			const Vec2 tangent = n.GetPerpendicular();
			jacobian.At(1, 0) = -tangent.x;
			jacobian.At(1, 1) = -tangent.y;
			jacobian.At(1, 2) = (-ra).Cross(tangent);

			jacobian.At(1, 3) = tangent.x;
			jacobian.At(1, 4) = tangent.y;
			jacobian.At(1, 5) = rb.Cross(tangent);
		}
		else
		{
			for (int i = 0; i < 6; ++i)
			{
				jacobian.At(1, i) = 0.0f;
			}
		}

		//// Compute the bias term using Baumgarte stabilization
		constexpr float beta = 0.2f;
		const float dot = Dot(pb - pa, n);
		const float C = fminf(0.0f, dot + 0.01f);  // positional error
		biasFactor = (beta / deltaT) * C + (e * vRelativeAlongNormal);
	}

	void NonPenetrationConstraint::Solve()
	{
		const Mat<6, 1> V = GetVelocities();
		const Mat<6, 6> invM = GetInverseMassMatrix();

		const Mat<6, 2> Jt = jacobian.Transpose();

		Mat<2, 1> rhs = jacobian * V * -1.0f;
		rhs.At(0, 0) -= biasFactor;

		const Mat<2, 2> lhs = jacobian * invM * Jt;
		Mat<2, 1> lambda = SolveGaussSeidel(lhs, rhs);

		if (_friction > 0.0f)
		{
			const float tempLambda = std::max(lambda.At(0, 0), 0.0f);
			const float maxFriction = tempLambda * _friction;
			const float lTangent = lambda.At(1, 0);
			lambda.At(1, 0) = std::clamp(lTangent, -maxFriction, maxFriction);
		}

		const Mat<6, 1> impulses = Jt * lambda;

		auto& aBody = (*bodies)[a];
		auto& bBody = (*bodies)[b];
		aBody.AddLinearImpulse(impulses.At(0, 0), impulses.At(1, 0));
		aBody.AddAngularImpulse(impulses.At(2, 0));
		bBody.AddLinearImpulse(impulses.At(3, 0), impulses.At(4, 0));
		bBody.AddAngularImpulse(impulses.At(5, 0));
	}

	void NonPenetrationConstraint::PostSolve()
	{
		// Maybe clamp values of cached lambda to reasonable limits
	}

	void JointConstraint::PreSolve(float deltaT)
	{
		//	C' = JV + b = 0,
		//	where C' = the first time derivative of the position constraint function
		//		  J  = the jacobian
		//		  V  = the velocities vector
		//		  b  = the bias (smoothing) term

		// How to calculate J in the case of joint constraint?
		// [2(ra - rb), 2(lra X (ra - rb)), 2(rb - ra), r (lrb X (rb - ra))] * [ va, wa, vb, wb]
		//
		// where: ra = anchor point in world space, when calculated based on the perspective of body a
		//		  rb = anchor point in world space, when calculated based on the perspective of body b
		//		  lra = anchor point in a's local space
		//		  lrb = anchor point in b's local space
		//		  X = cross product :P

		auto& aBody = (*bodies)[a];
		auto& bBody = (*bodies)[b];

		const Vec2 pa = aBody.LocalToWorldSpace(anchorA);
		const Vec2 pb = bBody.LocalToWorldSpace(anchorB);

		const Vec2 j1 = (pa - pb) * 2.0f;

		jacobian.At(0, 0) = j1.x;		// Jacobian coefficient for A's linear velocity x
		jacobian.At(0, 1) = j1.y;		// Jacobian coefficient for A's linear velocity y

		const Vec2 ra = pa - aBody.position;
		const Vec2 rb = pb - bBody.position;

		jacobian.At(0, 2) = ra.Cross(pa - pb) * 2.0f;	// Jacobian coefficient for A's angular velocity

		const Vec2 j3 = (pb - pa) * 2.0f;

		jacobian.At(0, 3) = j3.x;		// Jacobian coefficient for B's linear velocity x
		jacobian.At(0, 4) = j3.y;		// Jacobian coefficient for B's linear velocity y

		jacobian.At(0, 5) = rb.Cross(pb - pa) * 2.0f;	// Jacobian coefficient for B's angular velocity

		// Warm starting:
		const Mat<6, 1> Jt = jacobian.Transpose();
		const Mat<6, 1> impulses = Jt * cachedLambda.At(0, 0);

		aBody.AddLinearImpulse(impulses.At(0, 0), impulses.At(1, 0));
		aBody.AddAngularImpulse(impulses.At(2, 0));
		bBody.AddLinearImpulse(impulses.At(3, 0), impulses.At(4, 0));
		bBody.AddAngularImpulse(impulses.At(5, 0));

		// Compute the bias term using Baumgarte stabilization
		constexpr float beta = 0.1f;
		const float C = Dot(pb - pa, pb - pa);  // positional error
		biasFactor = (beta / deltaT) * C;
	}

	void JointConstraint::Solve()
	{
		const Mat<6, 1> V = GetVelocities();
		const Mat<6, 6> invM = GetInverseMassMatrix();

		const Mat<6, 1> Jt = jacobian.Transpose();

		Mat<1, 1> rhs = jacobian * V * -1.0f;
		rhs.At(0, 0) -= biasFactor;

		const Mat<1, 1> lhs = jacobian * invM * Jt;
		const Mat<1, 1> lambda = SolveGaussSeidel(lhs, rhs);
		cachedLambda.At(0, 0) += lambda.At(0, 0);

		const Mat<6, 1> impulses = Jt * lambda.At(0, 0);

		auto& aBody = (*bodies)[a];
		auto& bBody = (*bodies)[b];
		aBody.AddLinearImpulse(impulses.At(0, 0), impulses.At(1, 0));
		aBody.AddAngularImpulse(impulses.At(2, 0));
		bBody.AddLinearImpulse(impulses.At(3, 0), impulses.At(4, 0));
		bBody.AddAngularImpulse(impulses.At(5, 0));
	}

	void JointConstraint::PostSolve()
	{
	}


	Mat<6, 1> Constraint::GetVelocities() const
	{
		const auto& aBody = (*bodies)[a];
		const auto& bBody = (*bodies)[b];
		const auto aInv = aBody.GetInverseMassAndMomentOfInertia();
		const auto& va = aBody.linearVelocity + aBody._linearImpulses * aInv.x;

		const auto bInv = bBody.GetInverseMassAndMomentOfInertia();
		const auto& vb = bBody.linearVelocity + bBody._linearImpulses * bInv.x;

		return Mat<6, 1> {
			va.x,
				va.y,
				aBody.angularVelocity + aBody._angularImpulses * aInv.y,
				vb.x,
				vb.y,
				bBody.angularVelocity + bBody._angularImpulses * bInv.y
		};
	}

	Mat<6, 6> Constraint::GetInverseMassMatrix() const
	{
		const auto invMassA = (*bodies)[a].GetInverseMassAndMomentOfInertia();
		const auto invMassB = (*bodies)[b].GetInverseMassAndMomentOfInertia();

		return Mat<6, 6>
		{
			invMassA.x, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
				0.0f, invMassA.x, 0.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 0.0f, invMassA.y, 0.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 0.0f, invMassB.x, 0.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 0.0f, invMassB.x, 0.0f,
				0.0f, 0.0f, 0.0f, 0.0f, 0.0f, invMassB.y
		};
	}
#pragma endregion

#pragma region Collisions

	// If the maximum separation is still negative, that means that we found the
	// smallest of penetrations => i.e. the penetration with the smallest change
	// required to be fixed.
	float FindMaxSeparation(const Box* a, const Box* b,
		OUT Vec2& axis,						// reference face normal, of length 1
		OUT Vec2& point,					// the point that penetrates
		OUT int& referenceFaceVertex)		// the index of the first vertex, with CW winding order, of the reference face (of rigidbody A)
	{
		float separation = -INFINITY;
		for (int i = 0; i < 4; ++i)
		{
			const int nextIndex = (i + 1) % 4;
			const Vec2 edgeStart = (*a)[i];
			const Vec2 edgeEnd = (*a)[nextIndex];
			const Vec2 edgeSegment = edgeEnd - edgeStart;

			// Note that this assumes a CW winding order; the normal should point away from the body
			const Vec2 currentAxis = edgeSegment.GetPerpendicular().NormalizedSafe();

			float minA{ INFINITY };
			float maxA{ -INFINITY };

			for (int j = 0; j < 4; ++j)
			{
				const float projection = Dot((*a)[j] - edgeStart, currentAxis);
				minA = fminf(minA, projection);
				maxA = fmaxf(maxA, projection);
			}

			Vec2 minVertex;
			bool foundValidPointOnCurrentAxis{ false };

			float minB{ INFINITY };
			float maxB{ -INFINITY };

			for (int j = 0; j < 4; ++j)
			{
				const Vec2 p = (*b)[j];
				const float projection = Dot(p - edgeStart, currentAxis);

				if (minA < projection && projection < maxA)
				{
					if (projection < maxA && projection > minA && projection <= minB)
					{
						const auto d0 = Dot(p - edgeStart, edgeSegment);
						const auto d1 = Dot(p - edgeEnd, edgeSegment);

						if ((d0 > 0.0f) == (d1 < 0.0f))
						{
							foundValidPointOnCurrentAxis = true;
							minVertex = p;
						}
					}
				}

				minB = fminf(minB, projection);
				maxB = fmaxf(maxB, projection);
			}

			if (minB > separation)
			{
				separation = minB;
				axis = currentAxis;
				referenceFaceVertex = i;

				if (foundValidPointOnCurrentAxis)
				{
					point = minVertex;
				}
			}
		}

		return separation;
	}

	// On a box, find the edge where one of the edge vertex indices is
	// the passed in incident edgeVertex, and where the edge normal is
	// least aligned with the reference edge normal.
	int FindIncidentEdgeIndex(const Box& box, Vec2 referenceEdgeNormal)
	{
		float minAlignment = INFINITY;
		int incidentEdgeIndex = -1;

		for (int i = 0; i < 4; ++i)
		{
			int startIndex = i;
			int endIndex = (startIndex + 1) % 4;
			const auto edgeNormal = (box[endIndex] - box[startIndex]).GetPerpendicular().NormalizedSafe();
			const float alignment = Dot(edgeNormal, referenceEdgeNormal);

			if (alignment < minAlignment)
			{
				minAlignment = alignment;
				incidentEdgeIndex = startIndex;
			}
		}

		return incidentEdgeIndex;
	}

	bool IsColliding(const Box& box1,
		Rigidbody& rba,
		const Box& box2,
		Rigidbody& rbb,
		OUT std::vector<Contact>& contacts)
	{
		Vec2 axis1, point1;
		int referenceFaceVertexA;
		const float sepA =
			FindMaxSeparation(&box1, &box2,
				OUT axis1,
				OUT point1,
				OUT referenceFaceVertexA);

		if (sepA > 0.0f)
		{
			return false;
		}

		Vec2 axis2, point2;
		int referenceFaceVertexB;
		const float sepB =
			FindMaxSeparation(&box2, &box1,
				OUT axis2,
				OUT point2,
				OUT referenceFaceVertexB);

		if (sepB > 0.0f)
		{
			return false;
		}

		const auto& referenceBox = sepA > sepB ? box1 : box2;
		const auto& incidentBox = sepA > sepB ? box2 : box1;

		const int referenceStartIndex = sepA > sepB ? referenceFaceVertexA : referenceFaceVertexB;
		const int referenceEndIndex = (referenceStartIndex + 1) % 4;

		const Vec2 referenceEdge = referenceBox[referenceEndIndex] - referenceBox[referenceStartIndex];
		const Vec2 referenceNormal = referenceEdge.GetPerpendicular().NormalizedSafe();

		const int incidentEdgeStartIndex = FindIncidentEdgeIndex(incidentBox, referenceNormal);
		const int incidentEdgeEndIndex = (incidentEdgeStartIndex + 1) % 4;

		const Vec2 v0 = incidentBox[incidentEdgeStartIndex];
		const Vec2 v1 = incidentBox[incidentEdgeEndIndex];

		std::vector<Vec2> contactPoints{ v0, v1 };
		std::vector<Vec2> clippedPoints = contactPoints;

		for (int i = 0; i < 4; ++i)
		{
			if (i == referenceStartIndex)
			{
				continue;
			}

			const Vec2 c0 = referenceBox[i];
			const Vec2 c1 = referenceBox[(i + 1) % 4];
			int numClipped = referenceBox.ClipSegmentToLine(contactPoints, clippedPoints, c0, c1);
			if (numClipped < 2)
			{
				break;
			}

			contactPoints = clippedPoints;
		}

		const auto vRef = referenceBox[referenceStartIndex];
		for (const auto& vClip : clippedPoints)
		{
			float separation = Dot(vClip - vRef, referenceNormal);
			if (separation <= 0)
			{
				Contact contact;
				contact.a = &rba;
				contact.b = &rbb;
				contact.normal = referenceNormal;
				contact.start = vClip;
				contact.end = vClip + referenceNormal * -separation;
				if (sepB >= sepA)
				{
					std::swap(contact.start, contact.end);		// start -> end points are always from "a" to "b"
					contact.normal = -contact.normal;			// collision normal should always point from "a" to "b"
				}

				contacts.push_back(contact);
			}
		}

		return true;
	}

	bool IsColliding(const Circle& a, Rigidbody& rba, const Circle& b, Rigidbody& rbb, OUT std::vector<Contact>& contacts)
	{
		const float radiusA = a.radius;
		const float radiusB = b.radius;
		const Vec2 positionB = rbb.position;
		const Vec2 positionA = rba.position;
		const Vec2 ab = positionB - positionA;

		if (ab.SqrMagnitude() <= (radiusA + radiusB) * (radiusA + radiusB))
		{
			Contact contact;
			contact.a = &rba;
			contact.b = &rbb;

			contact.normal = ab;
			contact.normal.NormalizeSafe();

			contact.start = positionB - contact.normal * radiusB;
			contact.end = positionA + contact.normal * radiusA;
			contact.depth = (contact.end - contact.start).Magnitude();
			contacts.push_back(contact);

			return true;
		}

		return false;
	}

	int FindClosestEdgeIndex(const Box& box, const Circle& circle, Rigidbody& boxRB, Rigidbody& circleRB, OUT std::vector<Contact>& contacts)
	{
		float closestDistanceSquared = circle.radius * circle.radius;
		int closestEdgeIndex = -1;

		float closestDistanceSquared_absolute = INFINITY;
		int closestEdgeIndex_absolute = -1;
		// What's the start index of the edge that is closest to the circle centre,
		// _regardless_ of that distance being larger than the circle radius
		// (in case the polygon contains the circle centre)

		int pointVsEdgePosition = 0;
		// -1: point is closest to the starting point; projection doesn't overlap the edge 
		//  0: point is projected onto the edge; 
		//  1: point is closest to the end point; projection doesn't overlap the edge

		bool isCircleInsidePolygon{ true };
		const Vec2 circleOrigin = circleRB.position;
		Contact contact;
		contact.a = &boxRB;
		contact.b = &circleRB;

		for (int i = 0; i < 4; ++i)
		{
			const int j = (i + 1) % 4;
			const Vec2& edgeStart = box[i];
			const Vec2& edgeEnd = box[j];

			const Vec2 AP = circleOrigin - edgeStart;
			const Vec2 BP = circleOrigin - edgeEnd;

			const bool isProjectedOntoEdge = Dot(AP, edgeEnd - edgeStart) > 0.0f &&
				Dot(BP, edgeStart - edgeEnd) > 0.0f;

			const Vec2 edgeNormal = (edgeEnd - edgeStart).GetPerpendicular().NormalizedSafe();
			const float distanceToEdge = Dot(edgeNormal, AP);
			if (distanceToEdge > 0.0f)
			{
				isCircleInsidePolygon = false;
			}

			if (isProjectedOntoEdge)
			{
				const float dSquared = distanceToEdge * distanceToEdge;

				if (dSquared < closestDistanceSquared)
				{
					closestDistanceSquared = dSquared;
					closestEdgeIndex = i;
					pointVsEdgePosition = 0;

					contact.normal = edgeNormal;
				}

				if (dSquared < closestDistanceSquared_absolute)
				{
					closestDistanceSquared_absolute = dSquared;
					closestEdgeIndex_absolute = i;
					contact.normal = edgeNormal;
				}
			}
			else
			{
				const float APSquared = AP.SqrMagnitude();
				if (APSquared < closestDistanceSquared)
				{
					closestDistanceSquared = APSquared;
					closestEdgeIndex = i;
					pointVsEdgePosition = -1;
				}

				const float BPSquared = BP.SqrMagnitude();
				if (BPSquared < closestDistanceSquared)
				{
					closestDistanceSquared = BPSquared;
					closestEdgeIndex = i;
					pointVsEdgePosition = 1;
				}
			}
		}

		if (isCircleInsidePolygon && closestEdgeIndex_absolute >= 0)
		{
			const Vec2 edgeStart = box[closestEdgeIndex_absolute];
			const Vec2 edgeEnd = box[(closestEdgeIndex_absolute + 1) % 4];

			const Vec2 edge = edgeEnd - edgeStart;
			const Vec2 ens = edge.NormalizedSafe();
			const Vec2 AP = circleOrigin - edgeStart;
			const float dot = Dot(AP, ens);

			contact.end = edgeStart + (ens * dot);
			contact.depth = -Dot(contact.normal, AP) + circle.radius;
			contact.start = circleOrigin;

			contacts.push_back(contact);

			return closestEdgeIndex_absolute;
		}

		if (closestEdgeIndex >= 0)
		{
			const Vec2 edgeStart = box[closestEdgeIndex];
			const Vec2 edgeEnd = box[(closestEdgeIndex + 1) % 4];

			if (pointVsEdgePosition == -1)
			{
				contact.end = edgeStart;
				contact.depth = circle.radius - sqrtf(closestDistanceSquared);
				contact.normal = -(edgeStart - circleOrigin).NormalizedSafe();
				contact.start = circleOrigin - (contact.normal * circle.radius);
			}

			if (pointVsEdgePosition == 0)
			{
				const Vec2 edge = edgeEnd - edgeStart;
				const Vec2 edgeNormalized = edge.NormalizedSafe();
				const Vec2 AP = circleOrigin - edgeStart;
				const float edgeSectionLength = Dot(AP, edgeNormalized);

				contact.normal.NormalizeSafe();
				contact.end = edgeStart + (edgeNormalized * edgeSectionLength);
				contact.depth = circle.radius - Dot(contact.normal, AP);
				contact.start = contact.end - (contact.normal * contact.depth);
			}

			if (pointVsEdgePosition == 1)
			{
				contact.end = edgeEnd;
				contact.depth = circle.radius - sqrtf(closestDistanceSquared);
				contact.normal = (circleOrigin - edgeEnd).NormalizedSafe();
				contact.start = circleOrigin - (contact.normal * circle.radius);
			}
		}

		if (closestEdgeIndex > -1)
		{
			contacts.push_back(contact);
		}

		return closestEdgeIndex;
	}

	bool IsColliding(const Box& box, const Circle& circle, Rigidbody& boxRB, Rigidbody& circleRB, OUT std::vector<Contact>& contacts)
	{
		const int edgeStartIndex = FindClosestEdgeIndex(box, circle, boxRB, circleRB, OUT contacts);
		return edgeStartIndex >= 0;
	}

	// This business w/ didSwapAandB is a bit hacky :(
	// I think we can get rid of it once Contact stops using pointers
	bool IsColliding(Rigidbody& a, Rigidbody& b, OUT std::vector<Contact>& contacts, OUT bool& didSwapAandB)
	{
		const RigidbodyType typeA = a.GetType();
		const RigidbodyType typeB = b.GetType();

		if (typeA == RigidbodyType::Circle && typeB == RigidbodyType::Circle)
		{
			return IsColliding(std::get<Circle>(a.shape), a, std::get<Circle>(b.shape), b, OUT contacts);
		}

		if (typeA == RigidbodyType::Box && typeB == RigidbodyType::Box)
		{
			return IsColliding(std::get<Box>(a.shape), a, std::get<Box>(b.shape), b, OUT contacts);
		}

		if ((typeA == RigidbodyType::Box && typeB == RigidbodyType::Circle) ||
			(typeA == RigidbodyType::Circle && typeB == RigidbodyType::Box))
		{
			if (typeA == RigidbodyType::Box)
			{
				return IsColliding(std::get<Box>(a.shape), std::get<Circle>(b.shape), a, b, OUT contacts);
			}
			else
			{
				bool isColliding = IsColliding(std::get<Box>(b.shape), std::get<Circle>(a.shape), b, a, OUT contacts);
				didSwapAandB = isColliding;
				return isColliding;
			}
		}

		return false;
	}

#pragma endregion

#pragma region World

	struct World
	{
		World() = default;

		// Particles
		Unalmas::SlotMapKey				CreateParticle(const Vec2& position, float mass);
		Unalmas::SlotMapKey				CreateParticle(float x, float y, float mass);
		Unalmas::SlotMapKey				ConnectWithSpring(Unalmas::SlotMapKey a, Unalmas::SlotMapKey b, float lengthAtRest, float k);
		Unalmas::SlotMapKey				CreateParticleConstraint(ConstraintType type, const Unalmas::SlotMapKey a, const Unalmas::SlotMapKey b, float param);
		void							DestroyParticle(const Unalmas::SlotMapKey particle);
		Unalmas::SlotMap<Particle>&		GetParticles() { return _particles; }
		
		// Rigidbodies
		Rigidbody& GetRigidbody(const Unalmas::SlotMapKey& key) const
		{
			return _rigidbodies[key];
		}

		Unalmas::SlotMapKey					CreateCircle(float radius, float mass, const Vec2& position);
		Unalmas::SlotMapKey					CreateCircle(float radius, float mass, float x, float y);
		Unalmas::SlotMapKey					CreateBox(float width, float height, float mass, const Vec2& position);
		Unalmas::SlotMapKey					CreateBox(float width, float height, float mass, float x, float y);
		Unalmas::SlotMap<Rigidbody>&		GetRigidbodies() { return _rigidbodies; }

		// Constraints
		Unalmas::SlotMapKey					CreateJointConstraint(Unalmas::SlotMapKey a, Unalmas::SlotMapKey b, const Vec2& anchorPoint);

		void								Step(float deltaT);

		Vec2								Gravity{ 0.0f, -9.81f };
		float								FixedTimeStep{ 0.01f };
		float								SpringDrag{ 0.01f };

		std::vector<Contact>				collisions;
		Mat<1, 6>							jacobianToLog;
		Mat<6, 1>							impulseToLog;

	private:
		void								GenerateSpringForces();
		void								EnforceParticleDistanceConstraint(const ParticleConstraint& constraint);
		void								StepParticles(float deltaT);
		void								StepRigidbodies(float deltaT);
		void								DetectCollisions();
		void								ResolveCollisions();

		Unalmas::SlotMap<Particle>			_particles;
		Unalmas::SlotMap<Spring>			_springs;
		Unalmas::SlotMap<Rigidbody>			_rigidbodies;

		// ---- Constraints
		Unalmas::SlotMap<ParticleConstraint>		_particleConstraints;
		Unalmas::SlotMap<JointConstraint>			_jointConstraints;
		Unalmas::SlotMap<NonPenetrationConstraint>	_nonPenetrationConstraints;
	};

	Unalmas::SlotMapKey World::CreateJointConstraint(Unalmas::SlotMapKey a, Unalmas::SlotMapKey b, const Vec2& anchorPoint)
	{
		return _jointConstraints.Insert(JointConstraint(&_rigidbodies, a, b, anchorPoint));
	}

	void World::DetectCollisions()
	{
		static std::vector<Contact> contacts;

		collisions.clear();

		//TODO finally a per-frame arena allocator use case!
		//	   even with manifold caching
		_nonPenetrationConstraints.Clear();

		//TODO implement broadphase; spatial partitioning

		for (int i = 0; i < _rigidbodies.Size(); ++i)
		{
			for (int j = i + 1; j < _rigidbodies.Size(); ++j)
			{
				bool didSwapAB = false;
				if (IsColliding(_rigidbodies[i], _rigidbodies[j], OUT contacts, OUT didSwapAB))
				{
					const Unalmas::SlotMapKey a = _rigidbodies.GetKeyForIndex(didSwapAB ? j : i);
					const Unalmas::SlotMapKey b = _rigidbodies.GetKeyForIndex(didSwapAB ? i : j);
					for (const auto& c : contacts)
					{
						collisions.push_back(c);
						_nonPenetrationConstraints.Insert(NonPenetrationConstraint(
							&_rigidbodies,
							a,
							b,
							c));
					}

					contacts.clear();
				}
			}
		}
	}

	void World::ResolveCollisions()
	{
		for (auto& contact : collisions)
		{
			Rigidbody* const a = contact.a;
			Rigidbody* const b = contact.b;

			const bool isBstatic = b->_inverseMass <= 0.0f;
			const bool isAstatic = a->_inverseMass <= 0.0f;

			if (isBstatic && isAstatic)
			{
				continue;
			}

			bool isBox{ false };

			// TODO remove when not testing anymore
			if (a->GetType() == RigidbodyType::Box || b->GetType() == RigidbodyType::Box)
			{
				isBox = true;
			}

			// We apply position correction depending on the relative mass
			// of the two bodies:
			//
			// aAdjustment = depth * (mB / (mA + mB))
			// bAdjustment = depth * (mA / (mA + mB))
			//
			// But apparently we can express the above in terms of 1/m, like so:
			//
			// aAdjustment = (depth / (1/mA + 1/mB)) * (1 / mA)
			// bAdjustment = (depth / (1/mA + 1/mB)) * (1 / mB)
			//

			const float d = contact.depth / (a->_inverseMass + b->_inverseMass);
			const float aAdjustment = isBstatic ? contact.depth : d * a->_inverseMass;
			const float bAdjustment = isAstatic ? contact.depth : d * b->_inverseMass;

			a->_positionAdjustment -= contact.normal * aAdjustment;
			b->_positionAdjustment += contact.normal * bAdjustment;

			//TODO - ??? - should we apply _positionAdjustments immediately, now?
			//			   In the current setup, it would not matter, because the
			//				collision has already been calculated before.

			const float e = fmin(a->restitution, b->restitution);

			// Calculate relative velocity, considering both linear velocity of the body, and angular velocity
			// at the point of contact
			const Vec2 ra = contact.end - a->position;
			const Vec2 rb = contact.start - b->position;
			const Vec2 va = a->linearVelocity + Vec2{ -a->angularVelocity * ra.y, a->angularVelocity * ra.x };
			const Vec2 vb = b->linearVelocity + Vec2{ -b->angularVelocity * rb.y, b->angularVelocity * rb.x };
			const Vec2 vRelative = va - vb;

			// Calculate impulse along the collision normal
			const float raXn = ra.x * contact.normal.y - ra.y * contact.normal.x;
			const float rbXn = rb.x * contact.normal.y - rb.y * contact.normal.x;
			const float raI = (raXn * raXn) * a->_inverseMomentOfInertia;
			const float rbI = (rbXn * rbXn) * b->_inverseMomentOfInertia;

			const float impulseMagnitude = (-(1.0f + e) * Dot(vRelative, contact.normal)) / (a->_inverseMass + b->_inverseMass + raI + rbI);
			const Vec2 jNormal = contact.normal * impulseMagnitude;

			// Calculate impulse along the collision tangent
			const float f = fminf(a->surfaceFriction.y, b->surfaceFriction.y);
			const Vec2 collisionTangent = contact.normal.GetPerpendicular();
			const float raXt = ra.x * collisionTangent.y - ra.y * collisionTangent.x;
			const float rbXt = rb.x * collisionTangent.y - rb.y * collisionTangent.x;
			const float raIt = (raXt * raXt) * a->_inverseMomentOfInertia;
			const float rbIt = (rbXt * rbXt) * b->_inverseMomentOfInertia;
			const float tImpulseMagnitude = f * (-(1.0f + e) * Dot(vRelative, collisionTangent)) / (a->_inverseMass + b->_inverseMass + raIt + rbIt);
			const Vec2 jTangent = collisionTangent * tImpulseMagnitude;

			const Vec2 j = jNormal + jTangent;

			a->AddImpulseAtPoint(j, ra);
			b->AddImpulseAtPoint(-j, rb);
		}
	}

	Unalmas::SlotMapKey World::CreateCircle(float radius, float mass, const Vec2& position)
	{
		Unalmas::SlotMapKey key = _rigidbodies.Insert(Rigidbody(radius, mass));
		_rigidbodies[key].SetPosition(position);
		return key;
	}

	Unalmas::SlotMapKey World::CreateBox(float width, float height, float mass, const Vec2& position)
	{
		Unalmas::SlotMapKey key = _rigidbodies.Insert(Rigidbody(width, height, mass));
		auto& body = _rigidbodies[key];
		body.SetPosition(position);
		body.RefreshWorldSpaceTransform();

		return key;
	}

	Unalmas::SlotMapKey World::CreateCircle(float radius, float mass, float x, float y)
	{
		return CreateCircle(radius, mass, Vec2(x, y));
	}

	Unalmas::SlotMapKey World::CreateBox(float width, float height, float mass, float x, float y)
	{
		return CreateBox(width, height, mass, Vec2(x, y));
	}

	void World::Step(float deltaT)
	{
		StepParticles(deltaT);
		StepRigidbodies(deltaT);
	}

	void World::StepRigidbodies(float deltaT)
	{
		DetectCollisions();

		for (auto& rigidbody : _rigidbodies)
		{
			rigidbody.IntegrateForces(deltaT, Gravity);
		}

		for (auto& constraint : _jointConstraints)
		{
			constraint.PreSolve(deltaT);
		}

		for (auto& constraint : _nonPenetrationConstraints)
		{
			constraint.PreSolve(deltaT);
		}

		constexpr int constraintSolveIterations = 2;
		for (int i = 0; i < constraintSolveIterations; ++i)
		{
			for (auto& constraint : _jointConstraints)
			{
				constraint.Solve();
			}

			for (auto& constraint : _nonPenetrationConstraints)
			{
				constraint.Solve();
			}
		}

		//TODO postSolve
		// What should we do in postSolve?

		for (auto& rigidbody : _rigidbodies)
		{
			rigidbody.IntegrateVelocities(deltaT);
			rigidbody.RefreshWorldSpaceTransform();
		}
	}

	void World::StepParticles(float deltaT)
	{
		GenerateSpringForces();

		for (auto& particle : _particles)
		{
			particle.Update(deltaT, Gravity);
		}

		for (const auto& constraint : _particleConstraints)
		{
			switch (constraint.type)
			{
			case ConstraintType::ParticleDistance:
				EnforceParticleDistanceConstraint(constraint);
				break;
			default:
				break;
			}
		}
	}

	void World::EnforceParticleDistanceConstraint(const ParticleConstraint& constraint)
	{
		Particle& a = _particles[constraint.a];
		Particle& b = _particles[constraint.b];

		{
			const Vec2 ab = b.position - a.position;
			const float magnitude = ab.Magnitude();
			const float diff = magnitude - constraint.param;
			if (diff < 0.0f || 0.0f < diff)
			{
				Vec2 normal{ 0.0f, 1.0f };
				if (magnitude > 0.0f)
				{
					normal = ab / magnitude;
				}

				const bool isAStatic = a._inverseMass <= 0.0f;
				const bool isBStatic = b._inverseMass <= 0.0f;

				const float aDistanceFactor = isAStatic ? 0.0f : isBStatic ? 1.0f : 0.5f;
				const float bDistanceFactor = isBStatic ? 0.0f : isAStatic ? 1.0f : 0.5f;

				a.position += (normal * (diff * aDistanceFactor));
				b.position -= (normal * (diff * bDistanceFactor));
			}
		}
	}

	Unalmas::SlotMapKey World::ConnectWithSpring(const Unalmas::SlotMapKey a,
		const Unalmas::SlotMapKey b,
		float lengthAtRest,
		float k)
	{
		return _springs.Insert(Spring(a, b, &_particles, lengthAtRest, k));
	}

	void World::GenerateSpringForces()
	{
		for (auto& spring : _springs)
		{
			if (spring.k > 0.0f)
			{
				auto& a = spring.A();
				auto& b = spring.B();
				Vec2 springForce = GenerateSpringForce(a.position, b.position, spring.lengthAtRest, spring.k);

				a.AddForce(-springForce);
				b.AddForce(springForce);

				a.AddForce(GenerateParticleDragForce(a, SpringDrag));
				b.AddForce(GenerateParticleDragForce(b, SpringDrag));
			}
		}
	}

	Unalmas::SlotMapKey World::CreateParticle(const Vec2& position, float mass)
	{
		return CreateParticle(position.x, position.y, mass);
	}

	Unalmas::SlotMapKey World::CreateParticle(float x, float y, float mass)
	{
		return _particles.Insert(Particle(x, y, mass));
	}

	Unalmas::SlotMapKey World::CreateParticleConstraint(ConstraintType type,
		const Unalmas::SlotMapKey a,
		const Unalmas::SlotMapKey b, float param)
	{
		return _particleConstraints.Insert(ParticleConstraint(type, a, b, param));
	}

	void World::DestroyParticle(Unalmas::SlotMapKey particle)
	{
		_particles.Erase(particle);
	}

#pragma endregion
} // namespace Bebop